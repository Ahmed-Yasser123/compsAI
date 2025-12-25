from __future__ import annotations
import asyncio
import hashlib
import io
import json
import math
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, quote_plus
from urllib import robotparser

import httpx

# Optional/soft dependencies. The pipeline degrades gracefully if any are missing.
try:
    import requests  # sync HTTP for robots/bs4-friendly fetch
except Exception:
    requests = None  # type: ignore
try:
    from bs4 import BeautifulSoup  # HTML parsing
except Exception:
    BeautifulSoup = None  # type: ignore
try:
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
except Exception:
    Nominatim = None  # type: ignore
    RateLimiter = None  # type: ignore
try:
    from PIL import Image, ExifTags
except Exception:
    Image = None  # type: ignore
    ExifTags = None  # type: ignore
try:
    import pytesseract
except Exception:
    pytesseract = None  # type: ignore
try:
    import torch
    import torchvision.transforms as T
    from torchvision.models import resnet50, ResNet50_Weights
except Exception:
    torch = None  # type: ignore
    T = None  # type: ignore
    resnet50 = None  # type: ignore
    ResNet50_Weights = None  # type: ignore
try:
    import numpy as np
except Exception:
    np = None  # type: ignore

# ----------------------------
# Paths / Config
# ----------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
IMAGES_DIR = REPO_ROOT / "images"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = "AutoCalcRealEstateScraper/1.0 (+https://example.com) python-httpx"
REQUEST_TIMEOUT = 20.0
NETWORK_DELAY_SECONDS = 1.0

# Back-compat: previous stubs saved under scraper/output — we now save under /results
OUTPUT_DIR = RESULTS_DIR

# ----------------------------
# Data models
# ----------------------------

@dataclass
class PropertyDetails:
    source: str
    url: Optional[str]
    address: Optional[str]
    price: Optional[float]
    bedrooms: Optional[float]
    bathrooms: Optional[float]
    area_sqft: Optional[float]
    lot_size_sqft: Optional[float]
    year_built: Optional[int]
    property_type: Optional[str]
    taxes: Optional[str]
    description: Optional[str]
    listing_agent: Optional[str]
    image_urls: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ----------------------------
# Utils
# ----------------------------

def sanitize_filename(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", (text or "").strip())[:120] or "file"

def parse_float(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    text = text.replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def approx_equal(a: Optional[float], b: Optional[float], pct: float = 0.1) -> bool:
    if a is None or b is None:
        return False
    if a == 0:
        return False
    return abs(a - b) <= pct * max(abs(a), abs(b))

def ensure_list_unique(seq: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

async def polite_delay():
    await asyncio.sleep(NETWORK_DELAY_SECONDS)

def check_robots_allow(url: str) -> bool:
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        allowed = rp.can_fetch(USER_AGENT, url)
        if not allowed:
            allowed = rp.can_fetch("*", url)
        return bool(allowed)
    except Exception:
        # If robots cannot be fetched, be conservative
        return False

# ----------------------------
# Geocoding (geopy with rate limiting)
# ----------------------------

def _geocode_sync(address: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    if Nominatim is None or RateLimiter is None:
        return None, None, None
    geolocator = Nominatim(user_agent=USER_AGENT, timeout=REQUEST_TIMEOUT)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, swallow_exceptions=True)
    loc = geocode(address, addressdetails=True)
    if not loc:
        return None, None, None
    return float(loc.latitude), float(loc.longitude), loc.address

def _reverse_geocode_sync(lat: float, lon: float) -> Optional[str]:
    if Nominatim is None or RateLimiter is None:
        return None
    geolocator = Nominatim(user_agent=USER_AGENT, timeout=REQUEST_TIMEOUT)
    reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1, swallow_exceptions=True)
    loc = reverse((lat, lon), zoom=18, addressdetails=True)
    return loc.address if loc else None

async def geocode_address(address: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    return await asyncio.to_thread(_geocode_sync, address)

async def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    return await asyncio.to_thread(_reverse_geocode_sync, lat, lon)

def offset_latlon(lat: float, lon: float, meters_north: float, meters_east: float) -> Tuple[float, float]:
    d_lat = meters_north / 111_320.0
    d_lon = meters_east / (40075000.0 * math.cos(math.radians(lat)) / 360.0)
    return lat + d_lat, lon + d_lon

# ----------------------------
# Image processing
# ----------------------------

def _conv2(img: "np.ndarray", kernel: "np.ndarray") -> "np.ndarray":  # type: ignore[name-defined]
    ky, kx = kernel.shape
    py, px = ky // 2, kx // 2
    padded = np.pad(img, ((py, py), (px, px)), mode="symmetric")
    out = np.zeros_like(img, dtype=np.float32)
    k = np.flipud(np.fliplr(kernel)).astype(np.float32)
    H, W = img.shape
    for y in range(H):
        for x in range(W):
            region = padded[y:y + ky, x:x + kx]
            out[y, x] = float(np.sum(region * k))
    return out

def _extract_exif(img: "Image.Image") -> Dict[str, Any]:  # type: ignore[name-defined]
    exif: Dict[str, Any] = {}
    if ExifTags is None:
        return exif
    try:
        raw = img._getexif() or {}
        tag_map = {ExifTags.TAGS.get(k, str(k)): v for k, v in raw.items()}
        for key in ["DateTime", "Make", "Model", "LensModel", "FNumber", "ExposureTime", "ISOSpeedRatings"]:
            if key in tag_map:
                exif[key] = tag_map[key]
    except Exception:
        pass
    return exif

def _classify_room_label_to_scene(label: str) -> str:
    s = label.lower()
    if any(k in s for k in ["stove", "oven", "microwave", "refrigerator", "dishwasher", "kitchen"]):
        return "kitchen"
    if any(k in s for k in ["sofa", "couch", "television", "entertainment", "living"]):
        return "living_room"
    if any(k in s for k in ["bed", "bedroom", "bunk"]):
        return "bedroom"
    if any(k in s for k in ["toilet", "sink", "shower", "bathtub", "bath"]):
        return "bathroom"
    if any(k in s for k in ["house", "home", "building", "lawn", "yard", "roof", "facade"]):
        return "exterior"
    return "unknown"

_CLASSIFIER: Dict[str, Any] = {"model": None, "preprocess": None, "ready": False}

def _load_classifier():
    if _CLASSIFIER["ready"]:
        return _CLASSIFIER["model"], _CLASSIFIER["preprocess"]
    if torch is None or resnet50 is None or ResNet50_Weights is None or T is None:
        _CLASSIFIER["ready"] = True
        return None, None
    try:
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.eval()
        preprocess = weights.transforms()
        _CLASSIFIER["model"] = model
        _CLASSIFIER["preprocess"] = preprocess
        _CLASSIFIER["ready"] = True
        return model, preprocess
    except Exception:
        _CLASSIFIER["ready"] = True
        return None, None

def analyze_image_bytes(content: bytes) -> Optional[Dict[str, Any]]:
    if Image is None or np is None:
        return None
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        return None

    w, h = img.size
    arr = np.asarray(img, dtype=np.float32)
    gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2])

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    sharpness = None
    try:
        laplacian_kernel = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=np.float32)
        lap = _conv2(gray, laplacian_kernel)
        sharpness = float(np.var(lap))
    except Exception:
        sharpness = None

    # EXIF
    exif = _extract_exif(img)

    # OCR
    ocr_text = ""
    if pytesseract is not None:
        try:
            ocr_text = pytesseract.image_to_string(img).strip()
        except Exception:
            ocr_text = ""

    # Classification
    label = None
    scene = None
    model, preprocess = _load_classifier()
    if model is not None and preprocess is not None and torch is not None:
        try:
            tensor = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                logits = model(tensor)
            prob = torch.nn.functional.softmax(logits, dim=1)
            topk = torch.topk(prob, k=1, dim=1)
            idx_cls = int(topk.indices[0, 0])
            label = ResNet50_Weights.DEFAULT.meta["categories"][idx_cls]  # type: ignore[union-attr]
            scene = _classify_room_label_to_scene(label)
        except Exception:
            label = None
            scene = None

    return {
        "width": w,
        "height": h,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2) if sharpness is not None else None,
        "exif": exif,
        "ocr_text": ocr_text,
        "classification_label": label,
        "scene_type": scene,
    }

async def download_and_process_images(image_urls: List[str], property_id: str) -> List[Dict[str, Any]]:
    if not image_urls:
        return []
    out_dir = IMAGES_DIR / property_id
    out_dir.mkdir(parents=True, exist_ok=True)

    urls = ensure_list_unique([u for u in image_urls if re.search(r"\.(jpg|jpeg|png|webp)(\?|$)", u, re.I)])[:50]
    results: List[Dict[str, Any]] = []

    async def process(url: str, idx: int):
        # respect robots.txt for each image host
        if not check_robots_allow(url):
            return
        await polite_delay()
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
                r = await client.get(url)
                if r.status_code != 200 or "image" not in r.headers.get("content-type", ""):
                    return
                content = r.content
        except Exception:
            return

        # Analyze and save (do CPU-bound work off-thread)
        metrics = await asyncio.to_thread(analyze_image_bytes, content)
        if metrics is None:
            return

        # save as jpeg
        if Image is not None:
            try:
                img = Image.open(io.BytesIO(content)).convert("RGB")
                path = out_dir / f"{idx:03d}.jpg"
                await asyncio.to_thread(img.save, path, "JPEG", quality=90)
                metrics["saved_path"] = str(path)
            except Exception:
                metrics["saved_path"] = None

        metrics["url"] = url
        results.append(metrics)

    await asyncio.gather(*(process(u, i + 1) for i, u in enumerate(urls)))
    return results

# ----------------------------
# HTML parsing for public pages
# ----------------------------

def bs4_extract_common(html: str) -> Dict[str, Any]:
    if BeautifulSoup is None:
        return {}
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)

    def find_price():
        m = re.search(r"\$\s?([0-9][0-9,\.]+)", text)
        return parse_float(m.group(0)) if m else None

    def find_beds():
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:bed|beds|bd|bds)\b", text, re.I)
        return parse_float(m.group(1)) if m else None

    def find_baths():
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:bath|baths|ba|bths)\b", text, re.I)
        return parse_float(m.group(1)) if m else None

    def find_sqft():
        m = re.search(r"([0-9][0-9,\.]+)\s*(?:sq\s?ft|sqft|ft(?:\u00B2|2))\b", text, re.I)
        return parse_float(m.group(1)) if m else None

    def find_year():
        m = re.search(r"built\s*(?:in\s*)?(\d{4})", text, re.I)
        try:
            y = int(m.group(1)) if m else None
            if y and 1800 < y < 2100:
                return y
        except Exception:
            return None
        return None

    # Description meta
    desc = None
    md = soup.find("meta", attrs={"name": "description"})
    if md and md.get("content"):
        desc = md.get("content")

    # Image URLs (public only)
    images: List[str] = []
    for tag in soup.find_all("img"):
        src = tag.get("src") or tag.get("data-src")
        if src and re.search(r"\.(jpg|jpeg|png|webp)(\?|$)", src, re.I):
            if not re.search(r"(?i)(sprite|icon|logo|pixel)", src):
                images.append(src)

    return {
        "price": find_price(),
        "bedrooms": find_beds(),
        "bathrooms": find_baths(),
        "area_sqft": find_sqft(),
        "year_built": find_year(),
        "description": desc,
        "image_urls": list(dict.fromkeys(images))[:40],
    }

async def fetch_provider_public(source: str, address: str) -> Optional[PropertyDetails]:
    """
    Attempts a public, legal fetch of property data for a given source.
    Respects robots.txt and skips if disallowed or not publicly accessible.
    """
    if requests is None:
        return None

    base_urls = {
        "zillow": f"https://www.zillow.com/homes/{quote_plus(address)}_rb/",
        "redfin": f"https://www.redfin.com/stingray/do/location-autocomplete?location={quote_plus(address)}",
        "realtor": f"https://www.realtor.com/realestateandhomes-search/{quote_plus(address)}",
    }
    url = base_urls.get(source.lower())
    if not url:
        return None

    if not check_robots_allow(url):
        return None

    # Polite delay
    await polite_delay()

    try:
        resp = await asyncio.to_thread(requests.get, url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})
        if resp.status_code != 200:
            return None
        html = resp.text
    except Exception:
        return None

    fields = bs4_extract_common(html)
    pd = PropertyDetails(
        source=source,
        url=url,
        address=address,
        price=fields.get("price"),
        bedrooms=fields.get("bedrooms"),
        bathrooms=fields.get("bathrooms"),
        area_sqft=fields.get("area_sqft"),
        lot_size_sqft=None,
        year_built=fields.get("year_built"),
        property_type=None,
        taxes=None,
        description=fields.get("description"),
        listing_agent=None,
        image_urls=fields.get("image_urls", []),
    )
    return pd

async def consolidate_property(address: str) -> Dict[str, Any]:
    # Try public sources
    providers: List[PropertyDetails] = []
    for source in ["zillow", "redfin", "realtor"]:
        pd = await fetch_provider_public(source, address)
        if pd:
            providers.append(pd)

    def pick_first(attr: str):
        for p in providers:
            v = getattr(p, attr, None)
            if v not in (None, [], ""):
                return v
        return None

    image_urls: List[str] = []
    for p in providers:
        image_urls.extend(p.image_urls or [])
    image_urls = list(dict.fromkeys(image_urls))[:40]

    # Use address as seed for image dir id
    prop_id = hashlib.sha1((address or "").encode("utf-8")).hexdigest()[:12]
    images_info = await download_and_process_images(image_urls, prop_id)

    consolidated = {
        "source_summaries": [p.to_dict() for p in providers],
        "address": address,
        "price": pick_first("price"),
        "bedrooms": pick_first("bedrooms"),
        "bathrooms": pick_first("bathrooms"),
        "area_sqft": pick_first("area_sqft"),
        "year_built": pick_first("year_built"),
        "property_type": pick_first("property_type"),
        "lot_size_sqft": pick_first("lot_size_sqft"),
        "taxes": pick_first("taxes"),
        "description": pick_first("description"),
        "listing_agent": pick_first("listing_agent"),
        "image_urls": image_urls,
        "images": images_info,
        "images_dir": str((IMAGES_DIR / prop_id).resolve()),
    }
    return consolidated

# ----------------------------
# Comparable discovery (N, S, E, W; up to 2 each)
# ----------------------------

CARDINALS = [("N", 1, 0), ("S", -1, 0), ("E", 0, 1), ("W", 0, -1)]

def meters_distance(lat0: float, lon0: float, lat1: float, lon1: float) -> float:
    R = 6371000.0
    x = math.radians(lon1 - lon0) * math.cos(math.radians((lat0 + lat1) / 2))
    y = math.radians(lat1 - lat0)
    return math.sqrt(x * x + y * y) * R

async def find_nearby_comps(lat: float, lon: float, beds: Optional[float], baths: Optional[float], area: Optional[float]) -> List[Dict[str, Any]]:
    if lat is None or lon is None:
        return []
    offsets = [150, 300]  # meters
    per_dir: Dict[str, List[Dict[str, Any]]] = {d: [] for d, _, _ in CARDINALS}

    for dist in offsets:
        for dname, yn, xe in CARDINALS:
            lat2, lon2 = offset_latlon(lat, lon, yn * dist, xe * dist)
            addr2 = await reverse_geocode(lat2, lon2)
            if not addr2:
                continue

            comps: List[Dict[str, Any]] = []
            comp_consolidated = await consolidate_property(addr2)
            if comp_consolidated.get("source_summaries"):
                comps.append(comp_consolidated)

            # Filter by same beds/baths and +/-10% area when present
            for comp in comps:
                b = comp.get("bedrooms")
                ba = comp.get("bathrooms")
                a = comp.get("area_sqft")
                ok = True
                if beds is not None and b is not None and float(beds) != float(b):
                    ok = False
                if ok and baths is not None and ba is not None and float(baths) != float(ba):
                    ok = False
                if ok and area is not None and a is not None:
                    if not (0.9 * area <= a <= 1.1 * area):
                        ok = False
                if ok:
                    dist_m = meters_distance(lat, lon, lat2, lon2)
                    per_dir[dname].append({
                        "direction_bucket": dname,
                        "distance_m": round(dist_m, 1),
                        "property": comp,
                    })

            per_dir[dname] = per_dir[dname][:2]

    comps_out: List[Dict[str, Any]] = []
    for dname in ["N", "S", "E", "W"]:
        comps_out.extend(per_dir[dname])
    return comps_out

# ----------------------------
# Provider scraping (best-effort HTML only) retained for compatibility
# (used nowhere in the new flow but kept to minimize disruption)
# ----------------------------

@dataclass
class PropertyData:
    address: str
    price: Optional[float] = None
    bedrooms: Optional[float] = None
    bathrooms: Optional[float] = None
    squareFootage: Optional[float] = None
    yearBuilt: Optional[int] = None
    propertyType: Optional[str] = None
    hoaPerMonth: Optional[float] = None
    lotSizeSqft: Optional[float] = None
    ownerName: Optional[str] = None
    description: Optional[str] = None
    imageUrls: List[str] = None  # type: ignore
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["imageUrls"] = self.imageUrls or []
        return d

# Legacy regex for earlier parser
PRICE_RE = re.compile(r"\$[\s]*([0-9][0-9,\.]+)")
BEDS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:bed|beds|bd|bds)\b", re.I)
BATHS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:bath|baths|ba|bths)\b", re.I)
SQFT_RE = re.compile(r"([0-9][0-9,\.]+)\s*(?:sq\s?ft|sqft|ft\u00B2)\b", re.I)
YEAR_RE = re.compile(r"built\s*(?:in\s*)?(\d{4})", re.I)

# ----------------------------
# Public entrypoint (used by backend)
# ----------------------------

async def fetch_property_data(
    *,
    address: str,
    property_type: Optional[str] = None,
    bedrooms: Optional[int] = None,
    bathrooms: Optional[float] = None,
    square_footage: Optional[int] = None,
    radius_miles: float = 2.0,
) -> dict:
    """
    Full pipeline:
      - Geocode address
      - Fetch public data from Zillow/Redfin/Realtor (respect robots)
      - Download and analyze images (EXIF, OCR, classification)
      - Find nearby comps N/S/E/W (up to 2 each), repeat extraction
      - Save consolidated JSON to /results/<normalized_address>.json
      - Save images to /images/<property_id>/
    Returns minimal dict expected by backend + scrapeOutputPath.
    """
    # Geocode first to get normalized address
    lat, lon, normalized = await geocode_address(address)
    normalized_addr = normalized or address

    # Subject property
    subject = await consolidate_property(normalized_addr)

    # Override with caller-provided hints
    if property_type:
        subject["property_type"] = property_type
    if bedrooms is not None:
        subject["bedrooms"] = float(bedrooms)
    if bathrooms is not None:
        subject["bathrooms"] = float(bathrooms)
    if square_footage is not None:
        subject["area_sqft"] = float(square_footage)

    # Comps (cardinal directions)
    comps = []
    try:
        comps = await find_nearby_comps(
            lat=lat or 0.0, lon=lon or 0.0,
            beds=subject.get("bedrooms"),
            baths=subject.get("bathrooms"),
            area=subject.get("area_sqft"),
        ) if (lat is not None and lon is not None) else []
    except Exception:
        comps = []

    # Build final JSON object (prompt structure)
    result = {
        "query_address": address,
        "normalized_address": normalized_addr,
        "lat": lat,
        "lon": lon,
        "property": subject,
        "comparables": comps,
        "generated_at": int(time.time()),
        "notes": "Publicly accessible data only; robots.txt respected. Some fields may be missing.",
    }

    # Persist JSON
    out_name = sanitize_filename(normalized_addr) + ".json"
    out_path = OUTPUT_DIR / out_name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Minimal structure expected by backend (unchanged keys), map fields
    # propertyType, squareFootage are the backend schema's casing
    return {
        "address": normalized_addr,
        "propertyType": subject.get("property_type"),
        "bedrooms": subject.get("bedrooms"),
        "bathrooms": subject.get("bathrooms"),
        "squareFootage": subject.get("area_sqft"),
        "radiusMiles": radius_miles,
        "scrapeOutputPath": str(out_path),
    }
