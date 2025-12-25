
from __future__ import annotations
import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from typing import Tuple, TYPE_CHECKING, cast
from urllib.parse import quote_plus, urlparse, urljoin
from urllib import robotparser
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# Selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Image/HTTP handling
try:
    import requests
    from PIL import Image
except ImportError:
    requests = None
    Image = None
from io import BytesIO

# HTML parsing
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

# Configuration
REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = REPO_ROOT / "images"
RESULTS_DIR = REPO_ROOT / "results"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DEFAULT_PATH = REPO_ROOT / "scraper" / "imagescraper_cache.json"
OUTPUT_FILENAME = "imagescraper_results.json"  # single-file output each run

USER_AGENT = "AutoCalcImageScraper/1.0"
DEFAULT_WAIT = 15
POLITE_DELAY = 2.0  # seconds between requests (overridden by --throttle)


@dataclass
class PropertyData:
    """Comprehensive property information."""
    address: str
    url: Optional[str] = None
    source_site: Optional[str] = None
    
    # Core attributes
    price: Optional[float] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[float] = None
    square_footage: Optional[int] = None
    lot_size: Optional[str] = None
    year_built: Optional[int] = None
    property_type: Optional[str] = None
    
    # Financial
    monthly_hoa: Optional[float] = None
    annual_taxes: Optional[float] = None
    price_per_sqft: Optional[float] = None
    
    # Listing details
    listing_status: Optional[str] = None
    days_on_market: Optional[int] = None
    listing_agent: Optional[str] = None
    listing_agent_phone: Optional[str] = None
    owner_name: Optional[str] = None
    
    # Location
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    
    # Description
    description: Optional[str] = None
    
    # Images
    image_urls: Optional[List[str]] = None
    saved_images: Optional[List[str]] = None
    
    # Metadata
    scraped_at: Optional[str] = None
    direction_from_subject: Optional[str] = None
    distance_from_subject_meters: Optional[float] = None
    
    def __post_init__(self):
        if self.image_urls is None:
            self.image_urls = []
        if self.saved_images is None:
            self.saved_images = []


def build_driver(headless: bool = True, browser: Optional[str] = None):
    """Create a Selenium WebDriver using Selenium Manager."""
    last_err: Optional[Exception] = None

    def try_chrome():
        opts = ChromeOptions()
        if headless:
            opts.add_argument("--headless=new")
        opts.add_argument(f"--user-agent={USER_AGENT}")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_argument("--window-size=1920,1080")
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)
        return webdriver.Chrome(options=opts)

    def try_edge():
        opts = EdgeOptions()
        if headless:
            opts.add_argument("--headless=new")
        opts.add_argument(f"--user-agent={USER_AGENT}")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_argument("--window-size=1920,1080")
        return webdriver.Edge(options=opts)

    def try_firefox():
        opts = FirefoxOptions()
        if headless:
            opts.add_argument("-headless")
        try:
            opts.set_preference("general.useragent.override", USER_AGENT)
        except Exception:
            pass
        return webdriver.Firefox(options=opts)

    order = [browser.lower()] if browser else ["chrome", "edge", "firefox"]
    for b in order:
        try:
            if b == "chrome":
                return try_chrome()
            elif b == "edge":
                return try_edge()
            elif b == "firefox":
                return try_firefox()
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to initialize WebDriver. Tried {order}. Last error: {last_err}")


def robots_allow(url: str) -> bool:
    """Check if robots.txt allows scraping."""
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return bool(rp.can_fetch(USER_AGENT, url))
    except Exception:
        return True  # If we can't check, proceed cautiously


def make_search_url(site: str, address: str) -> str:
    """Generate search URL for the given site."""
    q = quote_plus(address)
    site = site.lower()
    
    if site == "zillow":
        return f"https://www.zillow.com/homes/{q}_rb/"
    elif site == "redfin":
        return f"https://www.redfin.com/stingray/do/location-autocomplete?location={q}"
    else:  # realtor
        return f"https://www.realtor.com/realestateandhomes-search/{q}"


def parse_float(text: Optional[str]) -> Optional[float]:
    """Extract float from text containing numbers and commas."""
    if not text:
        return None
    text = str(text).replace(",", "").replace("$", "")
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return None
    return None

def _sanitize_field(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    blacklist = [
        "save search", "more options", "manage rentals", "advertise", "get help", "sign in",
        "back to search", "home value", "cost calculator", "list your home", "find your next renter",
        "zillow rental manager", "plus"
    ]
    t = re.sub(r"\s+", " ", text.strip())
    tl = t.lower()
    if sum(1 for w in blacklist if w in tl) >= 2:
        return None
    for w in blacklist:
        t = re.sub(rf"(?i){re.escape(w)}", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t if len(t) >= 2 else None


def extract_property_details(driver, site: str, address: str) -> PropertyData:
    """Simplified & refined extraction for a property page."""
    prop = PropertyData(address=address, source_site=site, url=driver.current_url)
    prop.scraped_at = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        page_text = driver.find_element(By.TAG_NAME, "body").text
        page_html = driver.page_source
    except Exception:
        page_text = ""
        page_html = ""
    soup = BeautifulSoup(page_html, "html.parser") if BeautifulSoup else None
    # Price priority: Sold/List/Listing then fallback to largest valid dollar
    price_patterns = [
        r"Sold for \$\s*([0-9][0-9,\.]+)",
        r"List price:?\s*\$\s*([0-9][0-9,\.]+)",
        r"Listing price:?\s*\$\s*([0-9][0-9,\.]+)",
        r"\$\s*([0-9][0-9,\.]+)",
    ]
    for pat in price_patterns:
        m = re.search(pat, page_text, re.I)
        if m:
            val = parse_float(m.group(1))
            if val and val >= 10000:
                prop.price = val
                break
    if (prop.price is None) or (prop.price < 10000):
        all_prices = [parse_float(x) for x in re.findall(r"\$\s*([0-9][0-9,\.]+)", page_text)]
        all_prices = [p for p in all_prices if p and p >= 10000]
        if all_prices:
            prop.price = max(all_prices)
    # Beds / baths / sqft / year
    m_bed = re.search(r"(\d+)\s*(?:bed|beds|bd|bds)\b", page_text, re.I)
    if m_bed:
        prop.bedrooms = int(m_bed.group(1))
    m_bath = re.search(r"(\d+(?:\.\d+)?)\s*(?:bath|baths|ba)\b", page_text, re.I)
    if m_bath:
        prop.bathrooms = parse_float(m_bath.group(1))
    m_sqft = re.search(r"([0-9][0-9,\.]+)\s*(?:sq\.?\s?ft|sqft|square feet)", page_text, re.I)
    if m_sqft:
        sqv = parse_float(m_sqft.group(1))
        if sqv:
            prop.square_footage = int(sqv)
    m_year = re.search(r"built\s*(?:in\s*)?(\d{4})", page_text, re.I)
    if m_year:
        y = int(m_year.group(1))
        if 1800 < y <= 2030:
            prop.year_built = y
    # Lot / type / HOA / taxes
    m_lot = re.search(r"lot[:\s]+([0-9][0-9,\.]+\s*(?:acres?|sq\.?\s?ft|sqft))", page_text, re.I)
    if m_lot:
        prop.lot_size = m_lot.group(1).strip()
    m_type = re.search(r"(?:property type|type)[:\s]+([a-z\s]+)", page_text, re.I)
    if m_type:
        t_clean = _sanitize_field(m_type.group(1))
        if t_clean:
            prop.property_type = t_clean[:50]
    m_hoa = re.search(r"HOA[:\s]+\$?\s*([0-9][0-9,\.]+)(?:/mo|/month)?", page_text, re.I)
    if m_hoa:
        prop.monthly_hoa = parse_float(m_hoa.group(1))
    m_tax = re.search(r"(?:annual tax|property tax|taxes)[:\s]+\$?\s*([0-9][0-9,\.]+)", page_text, re.I)
    if m_tax:
        prop.annual_taxes = parse_float(m_tax.group(1))
    # Status / DOM
    m_status = re.search(r"status[:\s]+([a-z\s]+)", page_text, re.I)
    if m_status:
        st = _sanitize_field(m_status.group(1))
        if st:
            prop.listing_status = st[:30]
    m_dom = re.search(r"(\d+)\s*days?\s*on\s*(?:market|site)", page_text, re.I)
    if m_dom:
        prop.days_on_market = int(m_dom.group(1))
    # Agent / owner
    agent_patterns = [
        r"listing agent[:\s]+([^\n]{2,100})",
        r"listed by[:\s]+([^\n]{2,100})",
        r"agent[:\s]+([^\n]{2,100})",
        r"broker[:\s]+([^\n]{2,100})",
    ]
    for ap in agent_patterns:
        ma = re.search(ap, page_text, re.I)
        if ma:
            ag = _sanitize_field(ma.group(1))
            if ag:
                prop.listing_agent = ag[:100]
                break
    m_phone = re.search(r"(\(?\d{3}\)?[-\.\s]\d{3}[-\.\s]\d{4})", page_text)
    if m_phone:
        prop.listing_agent_phone = m_phone.group(1)
    m_owner = re.search(r"owner[:\s]+([a-z\s\.]+)", page_text, re.I)
    if m_owner:
        ow = _sanitize_field(m_owner.group(1))
        if ow:
            prop.owner_name = ow[:100]
    # Description
    if soup:
        for selector in [
            {"class": re.compile(r"description", re.I)},
            {"class": re.compile(r"remarks", re.I)},
            {"id": re.compile(r"description", re.I)},
        ]:
            de = soup.find("div", attrs=selector)  # type: ignore[arg-type]
            if de:
                prop.description = de.get_text(strip=True)[:1000]
                break
    # Zillow JSON fallback for numbers
    try:
        if site.lower() == "zillow":
            def num_from_html(pats: List[str]) -> Optional[float]:
                for pat in pats:
                    m = re.search(pat, page_html, re.I)
                    if m:
                        try:
                            return float(str(m.group(1)).replace(",", ""))
                        except Exception:
                            continue
                return None
            if (prop.price is None) or (prop.price < 10000):
                jp = num_from_html([
                    r'"currentPrice"\s*:\s*([0-9][0-9,\.]*)',
                    r'"price"\s*:\s*([0-9][0-9,\.]*)',
                    r'"homePrice"\s*:\s*([0-9][0-9,\.]*)',
                    r'"listPrice"\s*:\s*([0-9][0-9,\.]*)',
                    r'"soldPrice"\s*:\s*([0-9][0-9,\.]*)',
                ])
                if jp and jp >= 10000:
                    prop.price = jp
            if prop.bedrooms is None:
                bd = num_from_html([r'"bedrooms"\s*:\s*([0-9]+)', r'"beds"\s*:\s*([0-9]+)'])
                if bd is not None:
                    prop.bedrooms = int(bd)
            if prop.bathrooms is None:
                ba = num_from_html([r'"bathrooms"\s*:\s*([0-9]+(?:\.[0-9]+)?)', r'"baths"\s*:\s*([0-9]+(?:\.[0-9]+)?)'])
                if ba is not None:
                    prop.bathrooms = float(ba)
            if prop.square_footage is None:
                sf = num_from_html([
                    r'"livingArea(Value)?"\s*:\s*([0-9][0-9,\.]*)',
                    r'"livingArea"\s*:\s*([0-9][0-9,\.]*)',
                    r'"homeSize"\s*:\s*([0-9][0-9,\.]*)',
                ])
                if sf is not None:
                    prop.square_footage = int(sf)
    except Exception:
        pass
    # Images
    def _attr_to_str(v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, str):
                    return item
            return None
        if isinstance(v, bytes):
            try:
                return v.decode("utf-8", errors="ignore")
            except Exception:
                return None
        return str(v) if isinstance(v, str) else None
    if soup:
        imgs: List[str] = []
        for img in soup.find_all("img"):
            raw = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
            src = _attr_to_str(raw)
            if src and re.search(r"\.(jpg|jpeg|png|webp)", src, re.I):
                if not re.search(r"(logo|icon|sprite|avatar|pixel|1x1)", src, re.I):
                    imgs.append(urljoin(driver.current_url, src))
        prop.image_urls = list(dict.fromkeys(imgs))[:50]
    else:
        imgs = []
        for m in re.finditer(r"<img[^>]+(?:src|data-src|data-lazy-src)=['\"]([^'\"]+)['\"]", page_html, re.I):
            src = m.group(1)
            if src and re.search(r"\.(jpg|jpeg|png|webp)", src, re.I):
                if not re.search(r"(logo|icon|sprite|avatar|pixel|1x1)", src, re.I):
                    imgs.append(urljoin(driver.current_url, src))
        prop.image_urls = list(dict.fromkeys(imgs))[:50]
    if prop.price and prop.square_footage and prop.square_footage > 0 and not prop.price_per_sqft:
        prop.price_per_sqft = round(prop.price / prop.square_footage, 2)
    return prop


def download_images(
    prop: PropertyData,
    output_dir: Path,
    *,
    max_images: int = 40,
    max_workers: int = 4,
    throttle: float = 0.2,
) -> List[str]:
    """Download images concurrently and return list of saved paths."""
    if not prop.image_urls or not requests or not Image:
        return []

    urls = list(dict.fromkeys(prop.image_urls))[: max(0, max_images)]
    if not urls:
        return []

    # Use address-based naming directly in output_dir (no hashed subfolder)
    raw_address = prop.address or prop.url or "property"
    base_slug = re.sub(r"[^a-zA-Z0-9]+", "_", raw_address).strip("_")[:80]
    if not base_slug:
        base_slug = hashlib.md5(raw_address.encode()).hexdigest()[:12]
    site_part = (prop.source_site or "site").lower()
    # Ensure directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[str] = []

    def fetch_and_save(idx_url: Tuple[int, str]) -> Optional[str]:
        idx, url = idx_url
        try:
            time.sleep(throttle)
            headers = {"User-Agent": USER_AGENT}
            req = cast(Any, requests)
            resp = req.get(url, headers=headers, timeout=12)
            if resp.status_code != 200 or "image" not in resp.headers.get("content-type", ""):
                return None
            pil = cast(Any, Image)
            img = pil.open(BytesIO(resp.content))
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            filename = f"{base_slug}_{site_part}_{idx+1:03d}.jpg"
            save_path = output_dir / filename
            # If collision (same address rerun) overwrite is fine; optionally could timestamp.
            img.save(save_path, "JPEG", quality=90)
            return str(save_path)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(fetch_and_save, (i, u)) for i, u in enumerate(urls)]
        for fut in as_completed(futures):
            p = fut.result()
            if p:
                saved_paths.append(p)

    saved_paths.sort()
    return saved_paths


def scrape_property(
    driver,
    site: str,
    address: str,
    output_dir: Path,
    *,
    max_images: int = 40,
    throttle: float = 0.2,
    ignore_robots: bool = False,
) -> PropertyData:
    """Scrape a single property: navigate, extract details, download images."""
    print(f"\n  Scraping: {address}")
    url = make_search_url(site, address)
    if (not ignore_robots) and (not robots_allow(url)):
        print(f"  [SKIP] Robots.txt disallows: {url}")
        return PropertyData(address=address, source_site=site, url=url)
    try:
        driver.get(url)
        time.sleep(POLITE_DELAY)
        WebDriverWait(driver, DEFAULT_WAIT).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        prop = extract_property_details(driver, site, address)
        if prop.image_urls:
            print(f"  Downloading up to {min(max_images, len(prop.image_urls))} images...")
            prop.saved_images = download_images(prop, output_dir, max_images=max_images, throttle=throttle)
            print(f"  Saved {len(prop.saved_images)} images")
        return prop
    except Exception as e:
        print(f"  [ERROR] Scraping failed: {e}")
        return PropertyData(address=address, source_site=site, url=url)


def save_results(subject: PropertyData, output_path: Path, *, extras: Optional[Dict[str, Any]] = None):
    result: Dict[str, Any] = {
        "subject_property": asdict(subject),
        "metadata": {"scraped_at": time.strftime("%Y-%m-%d %H:%M:%S")},
    }
    if extras:
        for k, v in extras.items():
            if v is not None:
                result[k] = v
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to: {output_path}")
    return result


# -----------------------------
# Simple JSON cache utilities
# -----------------------------
def _normalize_address(addr: str) -> str:
    s = (addr or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _cache_key(site: str, address: str) -> str:
    return f"{(site or '').lower()}|{_normalize_address(address)}"


def load_cache(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception as e:
        print(f"[WARN] Failed to load cache {path}: {e}")
    return {}


def save_cache(path: Path, data: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] Failed to save cache {path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Advanced property scraper with image download and comparable discovery"
    )
    parser.add_argument("address", help="Property address to search")
    parser.add_argument("--site", choices=["zillow", "redfin", "realtor"],
                        default="realtor", help="Site to scrape (default: realtor)")
    parser.add_argument("--all-sites", action="store_true",
                        help="Scrape the address across all supported sites (realtor, zillow, redfin)")
    parser.add_argument("--separate-results", action="store_true",
                        help="When using --all-sites, save one JSON per site instead of a single aggregated JSON")
    parser.add_argument("--browser", choices=["chrome", "edge", "firefox"],
                       help="Browser to use (default: auto-detect)")
    parser.add_argument("--no-headless", action="store_true",
                       help="Show browser window (for debugging)")
    parser.add_argument("--output", help="Output JSON filename (default: auto-generated)")
    parser.add_argument("--max-images", type=int, default=40, help="Max images to download for subject (default 40)")
    # (Removed geocode/cardinal comparable arguments)
    parser.add_argument("--throttle", type=float, default=POLITE_DELAY, help="Delay seconds between requests (default 2.0)")
    parser.add_argument("--no-cache", action="store_true", help="Disable using/saving cache")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache and rescrape, then update cache")
    parser.add_argument("--cache-ttl-hours", type=int, default=168, help="Cache TTL in hours (default 168, i.e., 7 days)")
    parser.add_argument("--cache-file", default=str(CACHE_DEFAULT_PATH), help="Path to shared cache JSON file")
    parser.add_argument("--ignore-robots", action="store_true", help="Ignore robots.txt checks (use responsibly)")
    # Zillow comparable homes section extraction
    parser.add_argument("--zillow-comps", action="store_true", help="Extract comparable homes from Zillow's Comparable homes section instead of geocode cardinal directions (Zillow only)")
    parser.add_argument("--zillow-comps-limit", type=int, default=12, help="Max comparable homes to extract from Zillow section (default 12)")
    parser.add_argument("--zillow-comps-deep", action="store_true", help="After collecting Zillow comparable section cards, visit each comparable URL and extract full details (price, beds, baths, sqft, images, etc.)")
    parser.add_argument("--zillow-comps-max-images", type=int, default=5, help="Max images to download per deep Zillow comparable (default 5; set 0 to skip images)")
    # Cross-site enrichment of Zillow comparables
    parser.add_argument("--cross-site-comps", action="store_true", help="After scraping Zillow comparables, also scrape their addresses on other sites (realtor, redfin) for enrichment")
    parser.add_argument("--cross-site-sites", default="realtor,redfin", help="Comma-separated list of other sites to enrich comparables (default: realtor,redfin)")
    parser.add_argument("--cross-site-max-images", type=int, default=3, help="Max images per comparable when scraping on other sites (default 3; set 0 to skip images)")
    
    args = parser.parse_args()
    
    # Determine sites to scrape
    sites: List[str]
    if args.all_sites:
        # Default to Zillow first as requested
        sites = ["zillow", "realtor", "redfin"]
    else:
        sites = [args.site]

    # Create output filename(s)
    # Always single output file (like rentcast_site_scraper). Allow override via --output.
    output_file = RESULTS_DIR / (args.output if args.output else OUTPUT_FILENAME)
    
    print("=" * 70)
    print("PROPERTY IMAGE SCRAPER")
    print("=" * 70)
    print(f"Address: {args.address}")
    if len(sites) == 1:
        print(f"Site: {sites[0]}")
    else:
        print(f"Sites: {', '.join(sites)}")
        if args.separate_results:
            print("Saving separate JSON files per site.")
        else:
            print("Saving aggregated JSON across all sites.")
    print(f"Output: {output_file}")
    print("=" * 70)
    
    # Check dependencies
    missing = []
    if not requests:
        missing.append("requests")
    if not Image:
        missing.append("pillow")
    # Geocoding removed
    if not BeautifulSoup:
        missing.append("beautifulsoup4")
    
    if missing:
        print(f"\n[WARN] Missing optional dependencies: {', '.join(missing)}")
        print("Some features may be limited. Install with:")
        print(f"  pip install {' '.join(missing)}")
        print()
    
    # Initialize cache
    cache_path = Path(args.cache_file)
    cache = load_cache(cache_path) if not args.no_cache else {}

    print("\nInitializing browser...")
    driver = build_driver(headless=not args.no_headless, browser=args.browser)

    # ------------------------------
    # Helper: scroll & extract Zillow comparable homes
    # ------------------------------
    def fetch_zillow_comparables(driver, limit: int = 12) -> List[Dict[str, Any]]:
        comps: List[Dict[str, Any]] = []
        try:
            body_el = driver.find_element(By.TAG_NAME, "body")
            # Progressive scroll to trigger lazy load of comparable section/cards
            for i in range(22):
                driver.execute_script("window.scrollBy(0, Math.max(700, window.innerHeight * 0.9));")
                time.sleep(0.55)
                if i % 5 == 4:
                    driver.execute_script("window.scrollBy(0, -180);")
                txt_low = (body_el.text or "").lower()
                if "comparable homes" in txt_low or "similar homes" in txt_low:
                    break
            # First attempt: DOM anchors with /homedetails/ inside potential cards
            anchors = driver.find_elements(By.XPATH, "//a[contains(@href,'/homedetails/')]")
            seen = set()
            for a in anchors:
                if len(comps) >= limit:
                    break
                try:
                    href = a.get_attribute("href") or ""
                    if not href or href in seen:
                        continue
                    seen.add(href)
                    text_block = a.text or ""
                    # Zillow cards sometimes have text in parent; grab parent chain for enrichment
                    if (not text_block) or len(text_block.split()) < 4:
                        try:
                            parent = a.find_element(By.XPATH, "..")
                            parent_txt = parent.text or ""
                            if len(parent_txt) > len(text_block):
                                text_block = parent_txt
                        except Exception:
                            pass
                    # Extract fields
                    price_match = re.search(r"\$\s*[0-9][0-9,\.]+", text_block)
                    beds_match = re.search(r"(\d+)\s*(?:bd|beds?)\b", text_block, re.I)
                    baths_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:ba|baths?)\b", text_block, re.I)
                    sqft_match = re.search(r"([0-9][0-9,\.]*)\s*(?:sq\.?\s*ft|sqft|square feet)\b", text_block, re.I)
                    # Address heuristic: look for number start and a ZIP at end
                    addr_match = re.search(r"(\d{1,6}[^\n]{5,120}?\b[A-Z]{2}\s*\d{5})", text_block)
                    comp = {
                        "url": href,
                        "address": addr_match.group(1).strip() if addr_match else None,
                        "price": parse_float(price_match.group(0)) if price_match else None,
                        "beds": int(beds_match.group(1)) if beds_match else None,
                        "baths": float(baths_match.group(1)) if baths_match else None,
                        "sq_ft": int(parse_float(sqft_match.group(1)) or 0) if sqft_match else None,
                        "card_text": text_block[:1500],
                    }
                    # Require at least price or beds/baths to consider a valid comparable
                    if any([comp["price"], comp["beds"], comp["baths"], comp["sq_ft"], comp["address"]]):
                        comps.append(comp)
                except Exception:
                    continue
            # If still empty, fallback to HTML regex slice around Comparable/Similar homes header
            if not comps:
                html = driver.page_source
                m_hdr = re.search(r"(Comparable homes|Similar homes)", html, re.I)
                if m_hdr:
                    start = max(0, m_hdr.start() - 2000)
                    end = min(len(html), m_hdr.end() + 8000)
                    slice_html = html[start:end]
                else:
                    slice_html = html
                card_pattern = re.compile(r"<a[^>]+href=\"([^\"]+/homedetails/[^\"]+)\"[^>]*>([\s\S]*?)</a>", re.I)
                seen_urls = set()
                for a_href, a_inner in card_pattern.findall(slice_html)[: limit * 4]:
                    if len(comps) >= limit:
                        break
                    if a_href in seen_urls:
                        continue
                    seen_urls.add(a_href)
                    price_match = re.search(r"\$\s*[0-9][0-9,\.]+", a_inner)
                    beds_match = re.search(r"(\d+)\s*(?:bd|beds?)\b", a_inner, re.I)
                    baths_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:ba|baths?)\b", a_inner, re.I)
                    sqft_match = re.search(r"([0-9][0-9,\.]*)\s*(?:sq\.?\s*ft|sqft|square feet)\b", a_inner, re.I)
                    addr_match = re.search(r"(\d{1,6}[^<]{5,160}?\b[A-Z]{2}\s*\d{5})", a_inner)
                    comp = {
                        "url": urljoin("https://www.zillow.com/", a_href),
                        "address": addr_match.group(1).strip() if addr_match else None,
                        "price": parse_float(price_match.group(0)) if price_match else None,
                        "beds": int(beds_match.group(1)) if beds_match else None,
                        "baths": float(baths_match.group(1)) if baths_match else None,
                        "sq_ft": int(parse_float(sqft_match.group(1)) or 0) if sqft_match else None,
                        "raw_fragment": a_inner[:1800],
                    }
                    if any([comp["price"], comp["beds"], comp["baths"], comp["sq_ft"], comp["address"]]):
                        comps.append(comp)
            # Final dedupe by URL
            unique: Dict[str, Dict[str, Any]] = {}
            for c in comps:
                u = c.get("url") or ""
                if u and u not in unique:
                    unique[u] = c
            comps = list(unique.values())[:limit]
        except Exception as e:
            print(f"  [WARN] Zillow comps extraction failed: {e}")
        return comps

    try:
        now_ts = time.time()
        ttl_secs = max(0, int(args.cache_ttl_hours)) * 3600

        # Multi-site aggregation logic
        aggregated: Dict[str, Any] = {}
        # Simplified: no cardinal comparables

        for idx, site in enumerate(sites):
            print("\n" + "=" * 70)
            print(f"SUBJECT PROPERTY ({site})")
            print("=" * 70)

            # Cache key per site
            key = _cache_key(site, args.address)
            use_cache = False
            cached_result: Optional[Dict[str, Any]] = None
            if not args.no_cache and not args.refresh and key in cache:
                entry = cache.get(key) or {}
                ts = entry.get("timestamp")
                if isinstance(ts, (int, float)) and (ttl_secs == 0 or now_ts - ts <= ttl_secs):
                    cached_result = entry.get("result")
                    if isinstance(cached_result, dict):
                        use_cache = True
                        print(f"✓ Using cache for site '{site}' (age {int(now_ts - ts)}s)")

            if use_cache and cached_result:
                subject_dict = cached_result.get("subject_property") or {}
                metadata = cached_result.get("metadata") or {}
                # Ensure aggregated entry exists even when using cache
                aggregated[site] = {"subject_property": subject_dict, "metadata": metadata}
            else:
                subject = scrape_property(
                    driver,
                    site,
                    args.address,
                    IMAGES_DIR,
                    max_images=max(0, int(args.max_images)),
                    throttle=max(0.0, float(args.throttle)),
                    ignore_robots=bool(args.ignore_robots),
                )

                zillow_comps: List[Dict[str, Any]] = []
                zillow_comps_deep: List[Dict[str, Any]] = []
                # Zillow comparable homes section only
                if site == "zillow" and args.zillow_comps:
                    print("\n" + "=" * 70)
                    print("ZILLOW COMPARABLE HOMES SECTION")
                    print("=" * 70)
                    zillow_comps = fetch_zillow_comparables(driver, limit=max(1, int(args.zillow_comps_limit)))
                    print(f"  Extracted {len(zillow_comps)} comparable homes from section")
                    # Optional deep scrape of each comparable
                    zillow_comps_deep: List[Dict[str, Any]] = []
                    if args.zillow_comps_deep and zillow_comps:
                        print("\n" + "=" * 70)
                        print("ZILLOW COMPARABLE HOMES DEEP SCRAPE")
                        print("=" * 70)
                        for i, comp in enumerate(zillow_comps, start=1):
                            url = comp.get("url") or ""
                            if not url:
                                continue
                            try:
                                print(f"  [{i}/{len(zillow_comps)}] Visiting comparable: {url}")
                                driver.get(url)
                                time.sleep(max(0.2, float(args.throttle)))
                                WebDriverWait(driver, DEFAULT_WAIT).until(
                                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                                )
                                # Address fallback: use card address if extraction fails later
                                addr_for_prop = comp.get("address") or "Comparable Property"
                                deep_prop = extract_property_details(driver, "zillow", addr_for_prop)
                                # Limit images for deep comparables
                                if deep_prop.image_urls and int(args.zillow_comps_max_images) > 0:
                                    deep_prop.saved_images = download_images(
                                        deep_prop,
                                        IMAGES_DIR,
                                        max_images=max(0, int(args.zillow_comps_max_images)),
                                        throttle=max(0.0, float(args.throttle)),
                                    )
                                prop_dict = asdict(deep_prop)
                                # Merge card summary data for traceability
                                prop_dict["section_card"] = comp
                                zillow_comps_deep.append(prop_dict)
                            except Exception as e:
                                print(f"    [WARN] Deep scrape failed for {url}: {e}")
                                continue
                        print(f"  Deep scraped {len(zillow_comps_deep)} / {len(zillow_comps)} comparables")
                subject_dict = asdict(subject)
                metadata = {"scraped_at": time.strftime("%Y-%m-%d %H:%M:%S")}

                if not args.no_cache:
                    cache[key] = {"timestamp": now_ts, "result": {"subject_property": subject_dict, "metadata": metadata}}

                entry = {"subject_property": subject_dict, "metadata": metadata}
                if site == "zillow" and args.zillow_comps:
                    entry["zillow_comparables_section"] = zillow_comps
                    if args.zillow_comps_deep:
                        # zillow_comps_deep always defined when args.zillow_comps_deep evaluated earlier
                        entry["zillow_comparables_deep"] = zillow_comps_deep
                    # Optional cross-site enrichment
                    if args.cross_site_comps and zillow_comps:
                        print("\n" + "=" * 70)
                        print("ZILLOW COMPARABLES CROSS-SITE ENRICHMENT")
                        print("=" * 70)
                        other_sites = [s.strip().lower() for s in str(args.cross_site_sites).split(",") if s.strip()]
                        other_sites = [s for s in other_sites if s in ["realtor", "redfin"]]
                        if not other_sites:
                            print("  [WARN] No valid other sites specified for cross-site enrichment.")
                        enriched_list: List[Dict[str, Any]] = []
                        # Map deep data by URL for quick lookup
                        deep_map: Dict[str, Dict[str, Any]] = {}
                        if args.zillow_comps_deep and zillow_comps_deep:
                            for d in zillow_comps_deep:
                                u = d.get("url")
                                if u:
                                    deep_map[u] = d
                        for i, comp in enumerate(zillow_comps, start=1):
                            comp_addr = comp.get("address") or "Comparable Property"
                            comp_url = comp.get("url") or ""
                            print(f"  [{i}/{len(zillow_comps)}] Enriching comparable: {comp_addr}")
                            comp_entry: Dict[str, Any] = {"zillow_card": comp}
                            if comp_url and comp_url in deep_map:
                                comp_entry["zillow_deep"] = deep_map[comp_url]
                            other_site_data: Dict[str, Any] = {}
                            for osite in other_sites:
                                # Cache key per site/address
                                okey = _cache_key(osite, comp_addr)
                                o_use_cache = False
                                o_cached: Optional[Dict[str, Any]] = None
                                if not args.no_cache and not args.refresh:
                                    o_entry = cache.get(okey) or {}
                                    ots = o_entry.get("timestamp")
                                    if isinstance(ots, (int, float)) and (ttl_secs == 0 or now_ts - ots <= ttl_secs):
                                        o_res = o_entry.get("result")
                                        if isinstance(o_res, dict):
                                            o_use_cache = True
                                            o_cached = o_res
                                            print(f"      ✓ Using cache for comparable '{comp_addr}' on {osite}")
                                if o_use_cache and o_cached:
                                    other_site_data[osite] = o_cached
                                else:
                                    try:
                                        prop = scrape_property(
                                            driver,
                                            osite,
                                            comp_addr,
                                            IMAGES_DIR,
                                            max_images=max(0, int(args.cross_site_max_images)),
                                            throttle=max(0.0, float(args.throttle)),
                                            ignore_robots=bool(args.ignore_robots),
                                        )
                                        prop_dict = asdict(prop)
                                        meta = {"scraped_at": time.strftime("%Y-%m-%d %H:%M:%S")}
                                        other_site_data[osite] = {"subject_property": prop_dict, "metadata": meta}
                                        if not args.no_cache:
                                            cache[okey] = {"timestamp": now_ts, "result": other_site_data[osite]}
                                    except Exception as e:
                                        print(f"      [WARN] Enrichment failed on {osite}: {e}")
                            comp_entry["other_sites"] = other_site_data
                            enriched_list.append(comp_entry)
                        entry["zillow_comparables_cross_site"] = enriched_list
                aggregated[site] = entry

        # Save results
        print("\n" + "=" * 70)
        print("SAVING RESULTS")
        print("=" * 70)

        # Unified single-file output. Structure mirrors multi-site aggregated format always.
        final_payload = {"address": args.address, "sites": sites, "results": aggregated, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(final_payload, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved run results to single file: {output_file}")

        # Persist cache updates
        if not args.no_cache:
            save_cache(cache_path, cache)
            print(f"Cache updated: {cache_path}")

        print("\n" + "=" * 70)
        print("SCRAPING COMPLETE")
        print("=" * 70)
        if len(sites) == 1:
            site = sites[0]
            subj = aggregated[site]["subject_property"]
            print(f"Subject property ({site}): {subj.get('address')}")
            price = subj.get("price")
            beds = subj.get("bedrooms")
            baths = subj.get("bathrooms")
            imgs = subj.get("saved_images") or []
            print(f"  - Price: ${price:,.0f}" if isinstance(price, (int, float)) else "  - Price: N/A")
            print(f"  - Beds/Baths: {beds}/{baths}" if beds else "  - Beds/Baths: N/A")
            print(f"  - Images saved: {len(imgs)}")
        else:
            print(f"Sites scraped: {', '.join(sites)}")
            for site in sites:
                subj = aggregated[site]["subject_property"]
                price = subj.get("price")
                beds = subj.get("bedrooms")
                baths = subj.get("bathrooms")
                print(f"  [{site}] Price: ${price:,.0f}" if isinstance(price, (int, float)) else f"  [{site}] Price: N/A")
                print(f"          Beds/Baths: {beds}/{baths}" if beds else f"          Beds/Baths: N/A")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
