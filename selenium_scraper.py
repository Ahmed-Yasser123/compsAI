"""Headless Selenium scraper for public listing pages (best-effort).

IMPORTANT:
- Only scrape pages you have permission to access and that allow automated access.
- This script performs a robots.txt check and skips URLs that disallow crawling.
- Site HTML and selectors change frequently; the regex text parsing is a best-effort fallback.

Usage examples:
  # Interactive (prompts for addresses; optional property details)
  python selenium_scraper.py

  # Non-interactive, single or multiple addresses
  python selenium_scraper.py "8310 Haven St, Lenexa, KS 66219"
  python selenium_scraper.py "123 Main St, City, ST" "456 Oak Ave, Town, ST"

  # Choose site strategy (default: realtor)
  python selenium_scraper.py --site realtor "123 Main St, City, ST"
  python selenium_scraper.py --site zillow "123 Main St, City, ST"
  python selenium_scraper.py --site redfin "123 Main St, City, ST"

  # Add property details (applied to all addresses)
  python selenium_scraper.py --units 2 --buildings 1 --hoa-fee 0 --taxes 3200 --insurance 1800 "123 Main St, City, ST"

Output:
  - Writes selenium_results.json in this folder. For one address, a single object; for many, an array.

Dependencies:
    pip install selenium
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus, urlparse
from urllib import robotparser
 
# Selenium setup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

OUTPUT_FILE = Path(__file__).resolve().parent / "selenium_results.json"
USER_AGENT = "AutoCalcSelenium/1.0"
DEFAULT_WAIT = 15


def robots_allow(url: str) -> bool:
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        allowed = rp.can_fetch(USER_AGENT, url)
        return bool(allowed)
    except Exception:
        return False


def make_search_url(site: str, address: str) -> str:
    q = quote_plus(address)
    site = site.lower()
    if site == "realtor":
        return f"https://www.realtor.com/realestateandhomes-search/{q}"
    if site == "zillow":
        # For-sale by default; rentals would be /homes/for_rent
        return f"https://www.zillow.com/homes/{q}_rb/"
    if site == "redfin":
        return f"https://www.redfin.com/stingray/do/location-autocomplete?location={q}"
    # fallback: Google search (often blocked); we avoid this by default
    return f"https://www.realtor.com/realestateandhomes-search/{q}"


def extract_fields_from_text(text: str) -> Dict[str, Any]:
    def parse_float(s: Optional[str]) -> Optional[float]:
        if not s:
            return None
        s2 = s.replace(",", "")
        try:
            return float(re.search(r"-?\d+(?:\.\d+)?", s2).group(0))  # type: ignore[union-attr]
        except Exception:
            return None

    beds = None
    baths = None
    sqft = None
    price = None

    # Beds / Baths / Sqft patterns
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:bed|beds|bd|bds)\b", text, re.I)
    if m:
        beds = parse_float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:bath|baths|ba|bths)\b", text, re.I)
    if m:
        baths = parse_float(m.group(1))
    m = re.search(r"([0-9][0-9,\.]+)\s*(?:sq\s?ft|sqft|ft(?:\u00B2|2))\b", text, re.I)
    if m:
        sqft = parse_float(m.group(1))

    # Price/rent: first currency-like token
    m = re.search(r"\$\s*([0-9][0-9,\.]*)", text)
    if m:
        price = parse_float(m.group(1))

    return {
        "beds": int(beds) if beds is not None else None,
        "baths": float(baths) if baths is not None else None,
        "sq_ft": int(sqft) if sqft is not None else None,
        "estimated_rent": int(price) if price is not None else None,
    }


@dataclass
class Result:
    address: str
    beds: Optional[int] = None
    baths: Optional[float] = None
    sq_ft: Optional[int] = None
    estimated_rent: Optional[int] = None
    low_estimate: Optional[int] = None
    high_estimate: Optional[int] = None
    units: Optional[int] = None
    buildings: Optional[int] = None
    hoa_fee: Optional[float] = None
    taxes: Optional[float] = None
    insurance: Optional[float] = None
    source_site: Optional[str] = None
    url: Optional[str] = None


def build_driver(headless: bool = True, browser: Optional[str] = None):
    """Create a Selenium WebDriver using Selenium Manager.

    Tries Chrome, then Edge, then Firefox unless a specific browser is requested.
    This removes the need for webdriver-manager and avoids driver-not-installed errors.
    """
    last_err: Optional[Exception] = None

    def try_chrome():
        opts = ChromeOptions()
        if headless:
            opts.add_argument("--headless=new")
        opts.add_argument(f"--user-agent={USER_AGENT}")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--window-size=1280,1200")
        return webdriver.Chrome(options=opts)

    def try_edge():
        opts = EdgeOptions()
        # Edge uses Chromium flags
        if headless:
            opts.add_argument("--headless=new")
        opts.add_argument(f"--user-agent={USER_AGENT}")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--window-size=1280,1200")
        return webdriver.Edge(options=opts)

    def try_firefox():
        opts = FirefoxOptions()
        if headless:
            # Use argument form to avoid lint attr warning
            opts.add_argument("-headless")
        # UA override for Firefox (best-effort)
        try:
            opts.set_preference("general.useragent.override", USER_AGENT)
        except Exception:
            pass
        return webdriver.Firefox(options=opts)

    order = [browser.lower()] if isinstance(browser, str) and browser else ["chrome", "edge", "firefox"]
    for b in order:
        try:
            if b == "chrome":
                return try_chrome()
            if b == "edge":
                return try_edge()
            if b == "firefox":
                return try_firefox()
        except Exception as e:
            last_err = e
            continue
    # If we get here, all attempts failed
    raise RuntimeError(f"Failed to initialize a WebDriver. Tried {order}. Last error: {last_err}")


def scrape_address(site: str, address: str, headless: bool = True, *, browser: Optional[str] = None) -> Result:
    url = make_search_url(site, address)
    if not robots_allow(url):
        return Result(address=address, source_site=site, url=url)

    driver = build_driver(headless=headless, browser=browser)
    try:
        driver.get(url)
        try:
            WebDriverWait(driver, DEFAULT_WAIT).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
        except Exception:
            pass
        text = driver.find_element(By.TAG_NAME, "body").text
        fields = extract_fields_from_text(text)
        return Result(address=address, source_site=site, url=url, **fields)
    finally:
        driver.quit()


def save_results(results: List[Result]):
    if len(results) == 1:
        payload: Any = asdict(results[0])
    else:
        payload = [asdict(r) for r in results]
    OUTPUT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} record(s) to {OUTPUT_FILE}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Selenium scraper for public property pages (best-effort)")
    p.add_argument("addresses", nargs="*", help="Address strings (one or many). If omitted, interactive mode is used.")
    p.add_argument("--site", choices=["realtor", "zillow", "redfin"], default="realtor", help="Site strategy to use.")
    p.add_argument("--no-headless", action="store_true", help="Run Chrome with visible window.")
    p.add_argument("--browser", choices=["chrome", "edge", "firefox"], help="Browser to use. Defaults to trying chrome→edge→firefox.")
    # property details (optional; applied to all addresses)
    p.add_argument("--units", type=int, help="# of units")
    p.add_argument("--buildings", type=int, help="# of buildings")
    p.add_argument("--hoa-fee", dest="hoa_fee", type=float, help="HOA fee (0 if none)")
    p.add_argument("--taxes", type=float, help="Taxes per year")
    p.add_argument("--insurance", type=float, help="Insurance per year (units < 5 defaults to 1500 if omitted)")
    return p.parse_args(argv)


def interactive_flow():
    site = input("Site (realtor/zillow/redfin) [realtor]: ").strip().lower() or "realtor"
    headless = True
    addrs = input("Enter address (or multiple separated by commas/newlines): ").strip()
    addresses = [a.strip() for chunk in addrs.replace("\r", "").split("\n") for a in chunk.split(",") if a.strip()]
    if not addresses:
        print("At least one address is required.")
        sys.exit(1)

    units = input("Units (#) [blank to skip]: ").strip()
    buildings = input("Buildings (#) [blank to skip]: ").strip()
    hoa = input("HOA fee [blank to skip]: ").strip()
    taxes = input("Taxes per year [blank to skip]: ").strip()
    insurance = input("Insurance per year [blank to skip]: ").strip()

    args = {
        "site": site,
        "headless": headless,
        "addresses": addresses,
        "units": int(units) if units else None,
        "buildings": int(buildings) if buildings else None,
        "hoa_fee": float(hoa) if hoa else None,
        "taxes": float(taxes) if taxes else None,
        "insurance": float(insurance) if insurance else None,
    }
    run(**args)


def run(*, site: str, headless: bool, addresses: List[str], units: Optional[int], buildings: Optional[int], hoa_fee: Optional[float], taxes: Optional[float], insurance: Optional[float], browser: Optional[str] = None):
    # insurance default rule if units < 5 and no explicit insurance
    if units is not None and insurance is None:
        if units < 5:
            insurance = 1500.0
            print("[INFO] Insurance auto-set to 1500 (units < 5). Use --insurance to override.")
        else:
            print("[WARN] Units >= 5 and no insurance provided. Consider specifying --insurance.")

    results: List[Result] = []
    for i, addr in enumerate(addresses, 1):
        print(f"[{i}/{len(addresses)}] Scraping {site} for '{addr}'...")
        r = scrape_address(site, addr, headless=headless, browser=browser)
        # merge property details
        if units is not None: r.units = units
        if buildings is not None: r.buildings = buildings
        if hoa_fee is not None: r.hoa_fee = hoa_fee
        if taxes is not None: r.taxes = taxes
        if insurance is not None: r.insurance = insurance
        results.append(r)

    save_results(results)
    print("Done.")


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    if not args.addresses:
        interactive_flow()
    else:
        run(
            site=args.site,
            headless=not args.no_headless,
            addresses=[a.strip() for a in args.addresses if a.strip()],
            units=args.units,
            buildings=args.buildings,
            hoa_fee=args.hoa_fee,
            taxes=args.taxes,
            insurance=args.insurance,
            browser=args.browser,
        )
