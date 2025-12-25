from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import time

# Selenium imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchWindowException


OUTPUT_FILENAME = "rentcast_site_results.json"
DEFAULT_TIMEOUT = 25

CANDIDATE_URLS = [
    # Single target (RentCast app). We keep both forms for minor redirect variance.
    "https://app.rentcast.io/app",
    "https://app.rentcast.io/",
]


def build_driver(headless: bool = True) -> webdriver.Chrome:
    opts = webdriver.ChromeOptions()
    if headless:
        # Use new headless where supported
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--window-size=1366,768")
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/119.0 Safari/537.36"
    )
    # Selenium Manager will fetch the driver automatically
    driver = webdriver.Chrome(options=opts)
    return driver


def visible_elements(driver, by: str, value: str):
    els = driver.find_elements(by, value)
    return [e for e in els if e.is_displayed()]


def debug_capture(driver: webdriver.Chrome, stage: str, enabled: bool, screenshot: bool = True) -> None:
    """Capture HTML and optional screenshot for a given stage when enabled."""
    if not enabled:
        return
    try:
        debug_dir = Path(__file__).resolve().parent / "debug_pages"
        debug_dir.mkdir(exist_ok=True)
        ts = int(time.time())
        (debug_dir / f"stage_{ts}_{stage}.html").write_text(driver.page_source, encoding="utf-8")
        if screenshot:
            try:
                driver.save_screenshot(str(debug_dir / f"stage_{ts}_{stage}.png"))
            except Exception:
                pass
    except Exception:
        pass


def find_address_input(driver: webdriver.Chrome) -> Optional[Any]:
    # Try several selectors likely used by address inputs
    selectors = [
        (By.CSS_SELECTOR, 'input[placeholder*="Address" i]'),
        (By.CSS_SELECTOR, 'input[placeholder*="Search" i]'),
        (By.CSS_SELECTOR, 'input[placeholder*="Enter" i]'),
        (By.CSS_SELECTOR, 'input[aria-label*="Address" i]'),
        (By.CSS_SELECTOR, '[role="combobox"] input'),
        (By.CSS_SELECTOR, 'input[name*="address" i]'),
        (By.CSS_SELECTOR, 'input[type="search"]'),
        (By.CSS_SELECTOR, 'input[type="text"]'),
    ]
    for by, sel in selectors:
        for el in visible_elements(driver, by, sel):
            try:
                p = (el.get_attribute("placeholder") or "") + (el.get_attribute("name") or "")
                if "address" in p.lower() or el:
                    return el
            except Exception:
                continue
    return None


def click_possible_search(driver: webdriver.Chrome):
    # Try to click a button that likely triggers the estimate/search
    candidates = [
        (By.XPATH, "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'estimate')]") ,
        (By.XPATH, "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'search')]") ,
        (By.XPATH, "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'get rent')]") ,
        (By.XPATH, "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'get estimate')]") ,
        (By.XPATH, "//div[contains(@class,'autocomplete')]//li[1]"),
        (By.CSS_SELECTOR, "button[type='submit']"),
    ]
    for by, sel in candidates:
        try:
            for el in visible_elements(driver, by, sel):
                el.click()
                return True
        except Exception:
            continue
    return False


def select_first_autocomplete_option(driver: webdriver.Chrome) -> bool:
    # Best-effort: click first option in a listbox/menu often used by address autocompletes
    candidates = [
        (By.CSS_SELECTOR, "ul[role='listbox'] li[role='option']"),
        (By.CSS_SELECTOR, "[data-testid*='autocomplete'] li"),
        (By.XPATH, "//li[contains(@class,'option') or contains(@class,'item')][1]"),
    ]
    for by, sel in candidates:
        try:
            els = visible_elements(driver, by, sel)
            if els:
                els[0].click()
                return True
        except Exception:
            continue
    return False


def click_expand_sections(driver: webdriver.Chrome) -> bool:
    """Best-effort clicks on common 'expand details' toggles so additional fields (like sq ft) render."""
    patterns = [
        (By.XPATH, "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'details')]") ,
        (By.XPATH, "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'more info')]") ,
        (By.XPATH, "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'show more')]") ,
        (By.XPATH, "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'expand')]") ,
        (By.XPATH, "//a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'details')]") ,
        (By.XPATH, "//a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'show more')]") ,
    ]
    clicked = False
    for by, sel in patterns:
        try:
            for el in visible_elements(driver, by, sel):
                try:
                    el.click()
                    clicked = True
                except Exception:
                    continue
        except Exception:
            continue
    return clicked


def fill_property_filters(
    driver: webdriver.Chrome,
    *,
    property_type: Optional[str] = None,
    bedrooms: Optional[int] = None,
    bathrooms: Optional[float] = None,
    square_feet: Optional[int] = None,
    gentle: bool = False,
    selection_pause: float = 0.3,
    select_retries: int = 2,
) -> Dict[str, Any]:
    """Fill property filter fields (property type, beds, baths, sqft) if found.
    
    Args:
        driver: Selenium WebDriver instance
        property_type: Property type string (e.g., "Single Family", "Multi-Family", "Condo", etc.)
        bedrooms: Number of bedrooms
        bathrooms: Number of bathrooms (can be float like 2.5)
        square_feet: Square footage
        gentle: If True, use slower/more careful interactions
        selection_pause: Pause duration after selections (seconds)
        select_retries: Number of retry attempts for selections
    """
    pause = selection_pause if gentle else 0.2
    
    def safe_click(element):
        """Attempt click with retries and verification."""
        for attempt in range(select_retries):
            try:
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
                time.sleep(pause * 0.5)
                element.click()
                time.sleep(pause)
                return True
            except Exception as e:
                if attempt == select_retries - 1:
                    # Last attempt - try JS click
                    try:
                        driver.execute_script("arguments[0].click();", element)
                        time.sleep(pause)
                        return True
                    except Exception:
                        return False
                time.sleep(pause)
        return False
    
    # Accumulate actual values selected/displayed
    actual: Dict[str, Any] = {
        "actual_property_type": None,
        "actual_bedrooms": None,
        "actual_bathrooms": None,
        "actual_sq_ft": None,
    }

    # Property Type dropdown or custom select
    if property_type:
        print(f"  [FILTER] Setting Property Type: {property_type}")
        type_patterns = [
            (By.XPATH, "//select[contains(translate(@name, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'property') and contains(translate(@name, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'type')]"),
            (By.XPATH, "//select[contains(translate(@id, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'property') and contains(translate(@id, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'type')]"),
            (By.XPATH, "//label[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'property type')]//following-sibling::select"),
            (By.XPATH, "//label[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'property type')]//parent::*/select"),
            (By.CSS_SELECTOR, "select[name*='propertyType' i], select[id*='propertyType' i]"),
        ]
        
        for by, sel in type_patterns:
            try:
                type_selects = visible_elements(driver, by, sel)
                if type_selects:
                    select_el = type_selects[0]
                    from selenium.webdriver.support.ui import Select
                    select_obj = Select(select_el)
                    
                    # Try exact match first, then case-insensitive partial
                    matched = False
                    for option in select_obj.options:
                        opt_text = (option.text or "").strip()
                        if opt_text.lower() == property_type.lower():
                            safe_click(option)
                            matched = True
                            print(f"    ✓ Selected: {opt_text}")
                            actual["actual_property_type"] = opt_text
                            break
                    
                    if not matched:
                        for option in select_obj.options:
                            opt_text = (option.text or "").strip()
                            if property_type.lower() in opt_text.lower() or opt_text.lower() in property_type.lower():
                                safe_click(option)
                                print(f"    ✓ Selected: {opt_text}")
                                actual["actual_property_type"] = opt_text
                                break
                    break
            except Exception as e:
                continue
        # Specific fallback for Angular/Bootstrap dropdowns using `.dropdown-item.ng-star-inserted`
        if actual["actual_property_type"] is None:
            try:
                # Try opening the dropdown by clicking on any nearby label/container first
                label_divs = visible_elements(
                    driver,
                    By.XPATH,
                    "//label[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'property type')]"
                )
                if label_divs:
                    safe_click(label_divs[0])
                    time.sleep(pause)
                # Now look for dropdown items
                items = visible_elements(driver, By.CSS_SELECTOR, ".dropdown-item.ng-star-inserted")
                if not items:
                    items = visible_elements(driver, By.CSS_SELECTOR, ".dropdown-item")
                # Normalize text for robust matching (ignore punctuation/hyphens)
                def norm(s: str) -> str:
                    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
                target = norm(property_type)
                for it in items:
                    t = (it.text or "").strip()
                    if not t:
                        continue
                    if norm(t) == target or target in norm(t):
                        if safe_click(it):
                            actual["actual_property_type"] = t
                            print(f"    ✓ Selected dropdown-item: {t}")
                            break
            except Exception:
                pass
        # Fallback attempt for custom div-based selects
        if actual["actual_property_type"] is None:
            try:
                # Open a custom dropdown by clicking a label container
                label_divs = visible_elements(driver, By.XPATH, "//div[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'property type')]")
                if label_divs:
                    safe_click(label_divs[0])
                    time.sleep(pause)
                options = visible_elements(driver, By.XPATH, "//div[@role='option' or @role='listitem' or contains(@class,'option')]")
                for op in options:
                    t = (op.text or "").strip()
                    if not t:
                        continue
                    if t.lower() == property_type.lower() or property_type.lower() in t.lower():
                        if safe_click(op):
                            actual["actual_property_type"] = t
                            print(f"    ✓ Selected custom: {t}")
                            break
            except Exception:
                pass
        # Specialized fallback for RentCast app SPA (e.g., app.rentcast.io) using React dropdowns
        if actual["actual_property_type"] is None and "rentcast" in (driver.current_url or ""):
            try:
                # Normalize for matching hyphen variants (multi family / multi-family / multifamily)
                def norm(s: str) -> str:
                    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
                target_norm = norm(property_type)
                synonyms = {target_norm}
                if "multi" in target_norm and "family" in target_norm:
                    synonyms.update({norm("Multi-Family"), norm("Multifamily"), norm("Multi Family")})
                # Attempt to open any dropdown trigger containing 'Property Type'
                triggers = visible_elements(driver, By.XPATH, "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'property type')] | //div[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'property type')]")
                if triggers:
                    safe_click(triggers[0])
                    time.sleep(pause)
                # Query all potential menu/catalog items via JS to capture dynamic portals
                js = "return Array.from(document.querySelectorAll('[role=option], .MuiMenuItem-root, .dropdown-item, li, div')).map(e=>e.innerText.trim()).filter(t=>t)"
                texts = driver.execute_script(js) or []
                chosen_text = None
                for txt in texts:
                    n = norm(txt)
                    if n in synonyms or any(n.startswith(s) and len(n) - len(s) < 10 for s in synonyms):
                        # Click element whose text matches
                        try:
                            el = driver.find_element(By.XPATH, f"//*[normalize-space(.)='{txt}']")
                        except Exception:
                            el = None
                        if el and safe_click(el):
                            chosen_text = txt
                            break
                        # Fallback partial contains match
                        if not chosen_text:
                            try:
                                el2 = driver.find_element(By.XPATH, f"//*[contains(normalize-space(.), '{txt}')]")
                                if el2 and safe_click(el2):
                                    chosen_text = txt
                                    break
                            except Exception:
                                pass
                if chosen_text:
                    actual["actual_property_type"] = chosen_text
                    print(f"    ✓ Selected SPA dropdown item: {chosen_text}")
                else:
                    print("    [WARN] SPA dropdown property type selection failed; consider --wait-for-user to select manually.")
            except Exception:
                print("    [WARN] SPA property type fallback encountered an error.")
    
    # Bedrooms
    if bedrooms is not None:
        print(f"  [FILTER] Setting Bedrooms: {bedrooms}")
        bed_patterns = [
            (By.XPATH, "//select[contains(translate(@name, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'bed')]"),
            (By.XPATH, "//select[contains(translate(@id, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'bed')]"),
            (By.XPATH, "//label[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'bedroom')]//following-sibling::select"),
            (By.XPATH, "//label[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'bedroom')]//parent::*/select"),
            (By.CSS_SELECTOR, "select[name*='bed' i], select[id*='bed' i]"),
        ]
        
        for by, sel in bed_patterns:
            try:
                bed_selects = visible_elements(driver, by, sel)
                if bed_selects:
                    select_el = bed_selects[0]
                    from selenium.webdriver.support.ui import Select
                    select_obj = Select(select_el)
                    
                    bed_str = str(bedrooms)
                    matched = False
                    for option in select_obj.options:
                        opt_text = (option.text or "").strip()
                        opt_val = (option.get_attribute("value") or "").strip()
                        if bed_str in opt_text or bed_str == opt_val:
                            safe_click(option)
                            matched = True
                            print(f"    ✓ Selected: {opt_text}")
                            actual["actual_bedrooms"] = bed_str
                            break
                    
                    if not matched:
                        # Try selecting by value
                        try:
                            select_obj.select_by_value(bed_str)
                            print(f"    ✓ Selected bedrooms: {bed_str}")
                            actual["actual_bedrooms"] = bed_str
                        except Exception:
                            pass
                    break
            except Exception:
                continue
        if actual["actual_bedrooms"] is None:
            # Attempt custom dropdown
            try:
                bed_labels = visible_elements(driver, By.XPATH, "//div[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'bedrooms')]")
                if bed_labels:
                    safe_click(bed_labels[0])
                    time.sleep(pause)
                options = visible_elements(driver, By.XPATH, "//div[@role='option' or contains(@class,'option')]")
                for op in options:
                    t = (op.text or "").strip()
                    if not t:
                        continue
                    if t.startswith(str(bedrooms)):
                        if safe_click(op):
                            actual["actual_bedrooms"] = str(bedrooms)
                            print(f"    ✓ Selected custom bedrooms: {t}")
                            break
            except Exception:
                pass
        # SPA-specific rc-input-select handling (RentCast Angular components)
        if actual["actual_bedrooms"] is None and "rentcast" in (driver.current_url or ""):
            try:
                # Open rc-input-select with placeholder Bedrooms
                dd = visible_elements(driver, By.XPATH, "//rc-input-select[contains(translate(@placeholder,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'bedroom')]//button[contains(@class,'dropdown-toggle')]")
                if dd:
                    safe_click(dd[0])
                    time.sleep(pause)
                    # collect dropdown-item entries
                    items = visible_elements(driver, By.CSS_SELECTOR, ".dropdown-item") or visible_elements(driver, By.XPATH, "//button[contains(@class,'dropdown-item')]")
                    target_int = int(bedrooms)
                    for it in items:
                        txt = (it.text or "").strip()
                        norm_txt = re.sub(r"[^a-z0-9]+"," ", txt.lower()).strip()
                        # match patterns like '4 beds', '4 bed', '4'
                        if re.search(rf"^{target_int}(?:\s|$)", norm_txt) or re.search(rf"^{target_int}\s*bed", norm_txt):
                            if safe_click(it):
                                actual["actual_bedrooms"] = str(target_int)
                                print(f"    ✓ Selected SPA bedrooms: {txt}")
                                break
                    # Capture selected-value span if present
                    if actual["actual_bedrooms"] is None:
                        sel_span = visible_elements(driver, By.XPATH, "//rc-input-select[contains(translate(@placeholder,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'bedroom')]//span[contains(@class,'selected-value')]")
                        if sel_span:
                            sv = (sel_span[0].text or "").strip()
                            if sv:
                                actual["actual_bedrooms"] = re.sub(r"[^0-9]+","", sv) or sv
            except Exception:
                pass
    
    # Bathrooms
    if bathrooms is not None:
        print(f"  [FILTER] Setting Bathrooms: {bathrooms}")
        bath_patterns = [
            (By.XPATH, "//select[contains(translate(@name, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'bath')]"),
            (By.XPATH, "//select[contains(translate(@id, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'bath')]"),
            (By.XPATH, "//label[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'bathroom')]//following-sibling::select"),
            (By.XPATH, "//label[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'bathroom')]//parent::*/select"),
            (By.CSS_SELECTOR, "select[name*='bath' i], select[id*='bath' i]"),
        ]
        
        for by, sel in bath_patterns:
            try:
                bath_selects = visible_elements(driver, by, sel)
                if bath_selects:
                    select_el = bath_selects[0]
                    from selenium.webdriver.support.ui import Select
                    select_obj = Select(select_el)
                    
                    bath_str = str(bathrooms)
                    matched = False
                    for option in select_obj.options:
                        opt_text = (option.text or "").strip()
                        opt_val = (option.get_attribute("value") or "").strip()
                        if bath_str in opt_text or bath_str == opt_val:
                            safe_click(option)
                            matched = True
                            print(f"    ✓ Selected: {opt_text}")
                            actual["actual_bathrooms"] = bath_str
                            break
                    
                    if not matched:
                        try:
                            select_obj.select_by_value(bath_str)
                            print(f"    ✓ Selected bathrooms: {bath_str}")
                            actual["actual_bathrooms"] = bath_str
                        except Exception:
                            pass
                    break
            except Exception:
                continue
        if actual["actual_bathrooms"] is None:
            try:
                bath_labels = visible_elements(driver, By.XPATH, "//div[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'bathrooms')]")
                if bath_labels:
                    safe_click(bath_labels[0])
                    time.sleep(pause)
                options = visible_elements(driver, By.XPATH, "//div[@role='option' or contains(@class,'option')]")
                for op in options:
                    t = (op.text or "").strip()
                    if not t:
                        continue
                    if t.startswith(str(bathrooms)):
                        if safe_click(op):
                            actual["actual_bathrooms"] = str(bathrooms)
                            print(f"    ✓ Selected custom bathrooms: {t}")
                            break
            except Exception:
                pass
        # SPA-specific rc-input-select handling for Bathrooms
        if actual["actual_bathrooms"] is None and "rentcast" in (driver.current_url or ""):
            try:
                dd = visible_elements(driver, By.XPATH, "//rc-input-select[contains(translate(@placeholder,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'bathroom')]//button[contains(@class,'dropdown-toggle')]")
                if dd:
                    safe_click(dd[0])
                    time.sleep(pause)
                    items = visible_elements(driver, By.CSS_SELECTOR, ".dropdown-item") or visible_elements(driver, By.XPATH, "//button[contains(@class,'dropdown-item')]")
                    # Bathrooms may be float (e.g., 2.5). Normalize comparison.
                    target_str = str(bathrooms).rstrip("0").rstrip(".") if isinstance(bathrooms, float) else str(bathrooms)
                    for it in items:
                        txt = (it.text or "").strip()
                        norm_txt = re.sub(r"[^a-z0-9.+]+"," ", txt.lower()).strip()
                        if norm_txt.startswith(target_str.lower()):
                            if safe_click(it):
                                actual["actual_bathrooms"] = target_str
                                print(f"    ✓ Selected SPA bathrooms: {txt}")
                                break
                    if actual["actual_bathrooms"] is None:
                        sel_span = visible_elements(driver, By.XPATH, "//rc-input-select[contains(translate(@placeholder,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'bathroom')]//span[contains(@class,'selected-value')]")
                        if sel_span:
                            sv = (sel_span[0].text or "").strip()
                            if sv:
                                # Extract leading number pattern
                                m = re.search(r"(\d+(?:\.\d+)?)", sv)
                                actual["actual_bathrooms"] = m.group(1) if m else sv
            except Exception:
                pass
    
    # Square Feet (input field)
    if square_feet is not None:
        print(f"  [FILTER] Setting Square Feet: {square_feet}")
        sqft_patterns = [
            (By.XPATH, "//input[contains(translate(@name, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'sqft') or contains(translate(@name, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'square')]"),
            (By.XPATH, "//input[contains(translate(@id, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'sqft') or contains(translate(@id, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'square')]"),
            (By.XPATH, "//label[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'square') and contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'feet')]//following-sibling::input"),
            (By.XPATH, "//label[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'square') and contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'feet')]//parent::*/input"),
            (By.XPATH, "//label[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'sq ft')]//following-sibling::input"),
            (By.CSS_SELECTOR, "input[name*='sqft' i], input[id*='sqft' i], input[name*='squareFeet' i], input[id*='squareFeet' i]"),
        ]
        
        for by, sel in sqft_patterns:
            try:
                sqft_inputs = visible_elements(driver, by, sel)
                if sqft_inputs:
                    input_el = sqft_inputs[0]
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", input_el)
                    time.sleep(pause * 0.5)
                    input_el.clear()
                    input_el.send_keys(str(square_feet))
                    val_after = input_el.get_attribute("value")
                    print(f"    ✓ Entered: {square_feet} (displayed: {val_after})")
                    actual["actual_sq_ft"] = val_after
                    time.sleep(pause)
                    break
            except Exception:
                continue
        # React/Angular input fallback via JS dispatch if still unset
        if actual["actual_sq_ft"] is None and "rentcast" in (driver.current_url or ""):
            try:
                js_input = driver.execute_script("return document.querySelector('input[placeholder*=\"Sq\"], input[placeholder*=\"sq ft\"], input[name*=\"sq\"], input[name*=\"area\"]')")
                if js_input:
                    driver.execute_script("arguments[0].value=arguments[1]; arguments[0].dispatchEvent(new Event('input',{bubbles:true})); arguments[0].dispatchEvent(new Event('change',{bubbles:true}));", js_input, str(square_feet))
                    val_after = js_input.get_attribute("value")
                    if val_after:
                        actual["actual_sq_ft"] = val_after
                        print(f"    ✓ JS-dispatched SqFt: {val_after}")
            except Exception:
                pass

    return actual


def perform_login(driver: webdriver.Chrome, email: Optional[str], password: Optional[str], timeout: int) -> bool:
    """Attempt to log in if a login form is present and credentials are provided."""
    if not email or not password:
        return False
    wait = WebDriverWait(driver, timeout)
    try:
        # Look for typical auth inputs
        email_input = None
        password_input = None
        for el in driver.find_elements(By.CSS_SELECTOR, 'input[type="email"], input[name*="email" i]'):
            if el.is_displayed():
                email_input = el
                break
        for el in driver.find_elements(By.CSS_SELECTOR, 'input[type="password"], input[name*="password" i]'):
            if el.is_displayed():
                password_input = el
                break
        if not email_input or not password_input:
            return False
        email_input.clear(); email_input.send_keys(email)
        password_input.clear(); password_input.send_keys(password)
        # Click a sign in button
        for by, sel in [
            (By.XPATH, "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'sign in')]") ,
            (By.XPATH, "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'log in')]") ,
            (By.CSS_SELECTOR, "button[type='submit']"),
        ]:
            try:
                btns = visible_elements(driver, by, sel)
                if btns:
                    btns[0].click()
                    break
            except Exception:
                pass
        # Wait for navigation/app shell
        wait.until(lambda d: 'app' in d.current_url or 'dashboard' in d.current_url or find_address_input(d) is not None)
        return True
    except Exception:
        return False


def page_text(driver: webdriver.Chrome) -> str:
    # Get visible text content of the page (fallback to page_source if needed)
    try:
        script = "return document.body.innerText || document.body.textContent || ''"
        txt = driver.execute_script(script) or ""
        return re.sub(r"\s+", " ", txt).strip()
    except Exception:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(driver.page_source, "html.parser")
        return re.sub(r"\s+", " ", soup.get_text(" ")).strip()


def parse_fields_from_text(text: str) -> Dict[str, Any]:
    def pf(rex: re.Pattern[str]) -> Optional[float]:
        m = rex.search(text)
        if not m:
            return None
        s = (m.group(1) or m.group(0)).replace(",", "")
        m2 = re.search(r"-?\d+(?:\.\d+)?", s)
        if not m2:
            return None
        try:
            return float(m2.group(0))
        except Exception:
            return None

    # Monetary values
    # Try to bias toward 'rent' mentions near dollar amounts
    rent = pf(re.compile(r"(?:estimated?\s*)?rent[^\n$]{0,32}\$\s*([0-9][0-9,\.]*)\b", re.I)) or pf(re.compile(r"\$\s*([0-9][0-9,\.]*)\b"))

    # Beds / Baths / Sqft
    beds = pf(re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:bed|beds|bd|bds)\b", re.I))
    baths = pf(re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:bath|baths|ba|bths)\b", re.I))
    # Sqft with multiple variants
    sqft = (
        pf(re.compile(r"\b([0-9][0-9,\.]*)\s*(?:sq\.?\s*ft\.?|sq\s*ft|sqft|ft(?:\u00B2|2))\b", re.I))
        or pf(re.compile(r"\b([0-9][0-9,\.]*)\s*(?:square\s*feet)\b", re.I))
        or pf(re.compile(r"(?<!\$)\b([0-9][0-9,\.]*)\s*sf\b", re.I))
    )

    # Low/High hints
    low = pf(re.compile(r"(?:low|min|lower bound)[^\d$]{0,12}\$?\s*([0-9][0-9,\.]*)", re.I))
    high = pf(re.compile(r"(?:high|max|upper bound)[^\d$]{0,12}\$?\s*([0-9][0-9,\.]*)", re.I))

    # Synthesize if only rent present
    est = None
    if rent is not None and (low is None or high is None):
        low = low if low is not None else float(rent) * 0.9
        high = high if high is not None else float(rent) * 1.15
        est = (low + high) / 2.0
    elif low is not None and high is not None:
        est = (low + high) / 2.0

    return {
        "beds": int(beds) if beds is not None else None,
        "baths": float(baths) if baths is not None else None,
        "sq_ft": int(sqft) if sqft is not None else None,
        "estimated_rent": int(est) if est is not None else (int(rent) if rent is not None else None),
        "low_estimate": int(low) if low is not None else None,
        "high_estimate": int(high) if high is not None else None,
    }


def merge_fields(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = base.copy()
    for k, v in extra.items():
        if out.get(k) is None and v is not None:
            out[k] = v
    return out


def merge_prefer_right(base: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dicts, preferring non-None values from 'right'."""
    out = base.copy()
    for k, v in right.items():
        if v is not None:
            out[k] = v
    return out


def scroll_and_capture_text(driver: webdriver.Chrome, *, max_scrolls: int = 8, pause: float = 0.8) -> str:
    """Scrolls the page incrementally to trigger lazy loads, returns combined visible text at the end."""
    last_len = 0
    for i in range(max_scrolls):
        try:
            driver.execute_script("window.scrollBy(0, Math.max(300, window.innerHeight*0.8));")
        except Exception:
            pass
        time.sleep(pause)
        txt = page_text(driver)
        if len(txt) <= last_len and i >= 2:
            # try a slight upward scroll to trigger observers
            try:
                driver.execute_script("window.scrollBy(0, -200);")
            except Exception:
                pass
            time.sleep(pause/2)
            txt = page_text(driver)
        if len(txt) > last_len:
            last_len = len(txt)
        else:
            # No growth; consider done
            break
    return page_text(driver)


def parse_labeled_fields(text: str) -> Dict[str, Any]:
    """Extract values based on explicit labels to improve accuracy and avoid guessing."""
    def num(s: Optional[str]) -> Optional[int]:
        if not s:
            return None
        try:
            m = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
            return int(float(m.group(0))) if m else None
        except Exception:
            return None

    out: Dict[str, Any] = {"estimated_rent": None, "low_estimate": None, "high_estimate": None}

    # Rent range like: "$1,200 - $1,450" near words 'rent' or 'range'
    m_range = re.search(r"(?:estimated\s*)?(?:rent|range)[^$\n]{0,60}\$\s*([\d,\.]+)\s*[-–]\s*\$\s*([\d,\.]+)", text, re.I)
    if m_range:
        lo = num(m_range.group(1))
        hi = num(m_range.group(2))
        if lo is not None and hi is not None:
            out["low_estimate"], out["high_estimate"] = lo, hi

    # Explicit labels
    m_est = re.search(r"estimated\s*(?:monthly\s*)?rent[^$\n]{0,40}\$\s*([\d,\.]+)", text, re.I)
    if m_est:
        val = num(m_est.group(1))
        if val is not None:
            out["estimated_rent"] = val

    m_low = re.search(r"(?:low(?:\s*estimate)?|lower\s*bound)[^$\n]{0,40}\$\s*([\d,\.]+)", text, re.I)
    if m_low:
        val = num(m_low.group(1))
        if val is not None:
            out["low_estimate"] = val

    m_high = re.search(r"(?:high(?:\s*estimate)?|upper\s*bound)[^$\n]{0,40}\$\s*([\d,\.]+)", text, re.I)
    if m_high:
        val = num(m_high.group(1))
        if val is not None:
            out["high_estimate"] = val

    return out


def parse_json_fields(html: str) -> Dict[str, Any]:
    """Search page HTML for structured numeric fields like rentHigh/rentLow/highEstimate/lowEstimate.
    This is best-effort and prefers exact keys if present.
    """
    out: Dict[str, Any] = {"estimated_rent": None, "low_estimate": None, "high_estimate": None}

    def num_match(pattern: str) -> Optional[int]:
        m = re.search(pattern, html, re.I)
        if not m:
            return None
        try:
            return int(float(m.group(1).replace(",", "")))
        except Exception:
            return None

    # Try common keys
    # e.g., "rentHigh": 2345
    for key in ["rentHigh", "highEstimate", "rent_high", "high_estimate", "upperBound", "upper"]:
        if out["high_estimate"] is None:
            out["high_estimate"] = num_match(rf"\"{key}\"\s*:\s*([0-9][0-9,\.]*)")

    for key in ["rentLow", "lowEstimate", "rent_low", "low_estimate", "lowerBound", "lower"]:
        if out["low_estimate"] is None:
            out["low_estimate"] = num_match(rf"\"{key}\"\s*:\s*([0-9][0-9,\.]*)")

    for key in ["rent", "estimatedRent", "estimated_rent"]:
        if out["estimated_rent"] is None:
            out["estimated_rent"] = num_match(rf"\"{key}\"\s*:\s*([0-9][0-9,\.]*)")

    return out


def parse_dom_fields(driver) -> Dict[str, Any]:
    """Use DOM label proximity to find amounts for Estimated/Low/High and ranges.
    Best-effort and resilient to SPA structures. Chooses min() for low and max() for high.
    """
    out: Dict[str, Any] = {"estimated_rent": None, "low_estimate": None, "high_estimate": None}

    def vis(els):
        try:
            return [e for e in els if getattr(e, "is_displayed", lambda: False)()]
        except Exception:
            return []

    # RentCast-specific: extract from the main rent estimate card (rc-statistics-card-range)
    # This avoids picking up values from comparables tables below
    try:
        # Find the main rent card by its unique component tag
        rent_cards = driver.find_elements(By.CSS_SELECTOR, "rc-statistics-card-range")
        if rent_cards and rent_cards[0].is_displayed():
            card = rent_cards[0]
            card_html = card.get_attribute('innerHTML') or ""
            
            # Extract High Estimate from the right column structure
            # Pattern: <div class="col text-right">...<div class="font-weight-bold text-dark"> $3,960 </div>
            high_match = re.search(
                r'High Estimate.*?<div[^>]*class="col text-right"[^>]*>.*?<div[^>]*class="font-weight-bold[^"]*"[^>]*>\s*\$\s*([0-9][0-9,\.]+)',
                card_html,
                re.DOTALL | re.IGNORECASE
            )
            if high_match:
                try:
                    out["high_estimate"] = int(float(high_match.group(1).replace(',', '')))
                except Exception:
                    pass
            
            # Extract Low Estimate from the left column
            low_match = re.search(
                r'Low Estimate.*?<div[^>]*class="col text-left"[^>]*>.*?<div[^>]*class="font-weight-bold[^"]*"[^>]*>\s*\$\s*([0-9][0-9,\.]+)',
                card_html,
                re.DOTALL | re.IGNORECASE
            )
            if low_match:
                try:
                    out["low_estimate"] = int(float(low_match.group(1).replace(',', '')))
                except Exception:
                    pass
            
            # Extract Estimated Monthly Rent from the main display
            est_match = re.search(
                r'Estimated Monthly Rent.*?<div[^>]*class="display-3[^"]*"[^>]*>\s*\$\s*([0-9][0-9,\.]+)',
                card_html,
                re.DOTALL | re.IGNORECASE
            )
            if est_match:
                try:
                    out["estimated_rent"] = int(float(est_match.group(1).replace(',', '')))
                except Exception:
                    pass
            
            # If we got values from the card, return immediately (don't fall back to generic search)
            if out["high_estimate"] or out["low_estimate"] or out["estimated_rent"]:
                return out
    except Exception:
        pass

    # Fallback to generic DOM proximity search if card-specific extraction failed
    def amounts_near(el) -> list[int]:
        vals: list[int] = []
        current = el
        ancestors = []
        for _ in range(4):
            try:
                ancestors.append(current)
                current = current.find_element(By.XPATH, "..")
            except Exception:
                break
        for node in ancestors:
            try:
                candidates = node.find_elements(By.XPATH, ".//*[contains(., '$')]")[:30]
                for c in candidates:
                    t = c.text or ""
                    for m in re.findall(r"\$\s*([0-9][0-9,\.]*)", t):
                        try:
                            vals.append(int(float(m.replace(",", ""))))
                        except Exception:
                            pass
                if vals:
                    break
            except Exception:
                continue
        # Deduplicate, keep order
        seen = set()
        uniq: list[int] = []
        for v in vals:
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        return uniq

    labels = {
        "estimated": ["estimated monthly rent", "estimated rent", "monthly rent"],
        "low": ["low estimate", "lower bound", "low"],
        "high": ["high estimate", "upper bound", "high"],
        "range": ["rent range", "range"],
    }

    def find_for(terms: list[str]) -> list[Any]:
        results: list[Any] = []
        for term in terms:
            try:
                xpath = f"//*[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{term}')]"
                results.extend(vis(driver.find_elements(By.XPATH, xpath)))
            except Exception:
                continue
        return results

    # Estimated
    for el in find_for(labels["estimated"]):
        vals = amounts_near(el)
        if vals:
            out["estimated_rent"] = vals[0]
            break

    # High only from DOM: keep low from pre-existing (label/text/JSON) logic to avoid $0 contamination
    # (We intentionally do not set low_estimate here.)
    for el in find_for(labels["high"]):
        vals = amounts_near(el)
        if vals:
            try:
                out["high_estimate"] = max(vals)
            except Exception:
                out["high_estimate"] = vals[-1]
            break

    # Range fallback: only use for high (leave low to previous parsing layers)
    if out["high_estimate"] is None:
        for el in find_for(labels["range"]):
            vals = amounts_near(el)
            if len(vals) >= 2:
                hi = max(vals)
                out["high_estimate"] = hi
                break

    return out


def normalize_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    low = fields.get("low_estimate")
    high = fields.get("high_estimate")
    rent = fields.get("estimated_rent")
    # Ensure ordering if both present
    if isinstance(low, (int, float)) and isinstance(high, (int, float)) and low > high:
        low, high = high, low
    # Only compute rent from range if rent is missing
    if (not isinstance(rent, (int, float))) and isinstance(low, (int, float)) and isinstance(high, (int, float)):
        rent = int(round((float(low) + float(high)) / 2.0))
    fields["low_estimate"] = int(low) if isinstance(low, (int, float)) else None
    fields["high_estimate"] = int(high) if isinstance(high, (int, float)) else None
    fields["estimated_rent"] = int(rent) if isinstance(rent, (int, float)) else None
    return fields


def save_results(record: Dict[str, Any]):
    path = Path(__file__).resolve().parent / OUTPUT_FILENAME
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Saved results to {path}")


def try_one_url(
    driver: webdriver.Chrome,
    url: str,
    address: str,
    timeout: int,
    debug: bool = False,
    email: Optional[str] = None,
    password: Optional[str] = None,
    *,
    max_scrolls: int = 8,
    scroll_pause: float = 0.8,
    synthesize_range: bool = False,
    range_low_pct: float = 0.9,
    range_high_pct: float = 1.15,
    use_filters: bool = False,
    property_type: Optional[str] = None,
    bedrooms: Optional[int] = None,
    bathrooms: Optional[float] = None,
    square_feet: Optional[int] = None,
    gentle: bool = False,
    selection_pause: float = 0.3,
    select_retries: int = 2,
) -> Optional[Dict[str, Any]]:
    print(f"Visiting: {url}")
    driver.get(url)
    wait = WebDriverWait(driver, timeout)

    # Accept cookie banners if present (best-effort)
    for sel in [
        (By.XPATH, "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'accept')]") ,
        (By.XPATH, "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'agree')]") ,
        (By.XPATH, "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'ok')]") ,
    ]:
        try:
            for el in visible_elements(driver, *sel):
                el.click()
                break
        except Exception:
            pass

    # If a login form is present and creds provided, attempt login
    try:
        has_password = any(e.is_displayed() for e in driver.find_elements(By.CSS_SELECTOR, 'input[type="password"], input[name*="password" i]'))
    except Exception:
        has_password = False
    if has_password and (email and password):
        perform_login(driver, email, password, timeout)

    # Locate address input
    try:
        addr_input = wait.until(lambda d: find_address_input(d))
    except TimeoutException:
        addr_input = find_address_input(driver)

    # For logging which filters we actually apply
    applied_filters: Dict[str, Any] = {}

    if addr_input:
        addr_input.clear()
        addr_input.send_keys(address)
        addr_input.send_keys(Keys.ENTER)
        # Try choosing autocomplete first option (common in address fields)
        select_first_autocomplete_option(driver)
        debug_capture(driver, "after_address", debug or use_filters)
        
        # Fill property filters if requested
        if use_filters:
            print("\n[INFO] Filling property feature filters...")
            time.sleep(0.5)  # Brief pause to let form elements render
            actuals = fill_property_filters(
                driver,
                property_type=property_type,
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                square_feet=square_feet,
                gentle=gentle,
                selection_pause=selection_pause,
                select_retries=select_retries,
            )
            applied_filters = {
                "requested_property_type": property_type,
                "requested_bedrooms": bedrooms,
                "requested_bathrooms": bathrooms,
                "requested_sq_ft": square_feet,
                **actuals,
            }
            debug_capture(driver, "after_filters", debug or use_filters)
            print("[INFO] Filters filled.\n")
        
        # Try also clicking a search/estimate button, just in case
        click_possible_search(driver)
        debug_capture(driver, "after_search_click", debug or use_filters)
        # Expand sections that may reveal sqft or details
        click_expand_sections(driver)
        debug_capture(driver, "after_expand", debug or use_filters)
    else:
        # As a fallback, try site search if available
        click_possible_search(driver)
        click_expand_sections(driver)

    # Wait for some result text to render
    try:
        wait.until(lambda d: "$" in page_text(d) or re.search(r"rent\s*estimate|estimated\s*rent", page_text(d), re.I))
    except TimeoutException:
        if debug:
            print("Timed out waiting for results.")

    text = page_text(driver)
    debug_capture(driver, "before_parse", debug or use_filters)

    # Targeted DOM rent parsing (supplemental): look for elements with labels near dollar ranges
    def targeted_dom_rent() -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            # Rent range patterns like $1,200 - $1,450 inside same element
            rent_els = driver.find_elements(By.XPATH, "//*[contains(., '$') and (contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'rent') or contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'estimate'))]")
            for el in rent_els[:20]:
                t = el.text or ""
                m = re.search(r"\$\s*([0-9][0-9,\.]+)\s*[-–]\s*\$\s*([0-9][0-9,\.]+)", t)
                if m:
                    try:
                        lo = int(float(m.group(1).replace(',', '')))
                        hi = int(float(m.group(2).replace(',', '')))
                        out['low_estimate'] = lo
                        out['high_estimate'] = hi
                        out['estimated_rent'] = int(round((lo + hi) / 2.0))
                        break
                    except Exception:
                        continue
                # Single rent value with label
                m2 = re.search(r"(?:rent|estimate)[^$]{0,20}\$\s*([0-9][0-9,\.]+)", t, re.I)
                if m2 and 'estimated_rent' not in out:
                    try:
                        val = int(float(m2.group(1).replace(',', '')))
                        out['estimated_rent'] = val
                    except Exception:
                        pass
        except Exception:
            pass
        return out

    supplemental_dom = targeted_dom_rent()
    debug_capture(driver, "after_dom_target", debug or use_filters)

    if debug:
        debug_dir = Path(__file__).resolve().parent / "debug_pages"
        debug_dir.mkdir(exist_ok=True)
        (debug_dir / "rentcast_last.html").write_text(driver.page_source, encoding="utf-8")
        try:
            driver.save_screenshot(str(debug_dir / "rentcast_last.png"))
        except Exception:
            pass

    fields = parse_fields_from_text(text)
    # Enrichment pass: scroll to load lazy sections and parse again
    # Scroll to load more UI and try expanding sections again mid-way
    scrolled_text = scroll_and_capture_text(driver, max_scrolls=max_scrolls, pause=scroll_pause)
    try:
        click_expand_sections(driver)
    except Exception:
        pass
    # Re-capture after expanding
    scrolled_text = page_text(driver)
    fields2 = parse_fields_from_text(scrolled_text)
    fields = merge_fields(fields, fields2)
    # Prefer labeled (more accurate) values where present
    labeled = parse_labeled_fields(scrolled_text)
    fields = merge_prefer_right(fields, labeled)
    # Use DOM proximity-based parsing for low/high/estimated
    try:
        dom_vals = parse_dom_fields(driver)
        fields = merge_prefer_right(fields, dom_vals)
    except Exception:
        pass
    # Try extracting from JSON-like structures in the HTML for precise values
    json_vals = parse_json_fields(driver.page_source)
    fields = merge_prefer_right(fields, json_vals)
    # If low and high are identical and no explicit range detected in text, drop high to avoid misleading duplicates
    had_labeled_high = False
    try:
        had_labeled_high = (
            labeled.get("high_estimate") is not None
            or (locals().get("dom_vals") or {}).get("high_estimate") is not None  # type: ignore[attr-defined]
            or json_vals.get("high_estimate") is not None
        )
        if (
            isinstance(fields.get("low_estimate"), int)
            and isinstance(fields.get("high_estimate"), int)
            and fields["low_estimate"] == fields["high_estimate"]
            and not had_labeled_high
            and not re.search(r"(?:rent|range)[^$\n]{0,60}\$\s*[\d,\.]+\s*[-–]\s*\$\s*[\d,\.]+", scrolled_text, re.I)
        ):
            fields["high_estimate"] = None
    except Exception:
        pass
    # If synthesis is enabled and we still lack a meaningful high, compute from rent
    if synthesize_range and fields.get("estimated_rent") is not None:
        has_explicit_range = bool(re.search(r"(?:rent|range)[^$\n]{0,60}\$\s*[\d,\.]+\s*[-–]\s*\$\s*[\d,\.]+", scrolled_text, re.I))
        if (fields.get("high_estimate") is None) or (
            isinstance(fields.get("low_estimate"), int)
            and isinstance(fields.get("high_estimate"), int)
            and fields["low_estimate"] == fields["high_estimate"]
            and not had_labeled_high
            and not has_explicit_range
        ):
            r = float(fields["estimated_rent"])  # type: ignore
            # Only set missing pieces; if low exists, keep it
            if fields.get("low_estimate") is None:
                fields["low_estimate"] = int(round(r * range_low_pct))
            # Only set high when missing (do not override explicit lower high)
            computed_high = int(round(r * range_high_pct))
            if fields.get("high_estimate") is None:
                fields["high_estimate"] = computed_high
    # Optionally synthesize range if missing but rent present
    if synthesize_range and fields.get("estimated_rent") is not None and (fields.get("low_estimate") is None and fields.get("high_estimate") is None):
        r = float(fields["estimated_rent"])  # type: ignore
        fields["low_estimate"] = int(round(r * range_low_pct))
        fields["high_estimate"] = int(round(r * range_high_pct))
    # Merge supplemental DOM rent parsing preferring existing explicit range if both present
    if supplemental_dom:
        for k, v in supplemental_dom.items():
            if fields.get(k) is None and v is not None:
                fields[k] = v
    fields = normalize_fields(fields)
    # If all key fields are None, consider this a failure for this URL
    if all(fields.get(k) is None for k in ["estimated_rent", "low_estimate", "high_estimate", "beds", "baths", "sq_ft"]):
        return None

    # Attach applied filter metadata so caller can include it in final record
    if use_filters:
        fields['filter_metadata'] = applied_filters
    return fields


def run(
    address: str,
    *,
    headless: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
    debug: bool = False,
    wait_for_user: bool = False,
    email: Optional[str] = None,
    password: Optional[str] = None,
    max_scrolls: int = 8,
    scroll_pause: float = 0.8,
    synthesize_range: bool = False,
    range_low_pct: float = 0.9,
    range_high_pct: float = 1.15,
    use_filters: bool = False,
    property_type: Optional[str] = None,
    bedrooms: Optional[int] = None,
    bathrooms: Optional[float] = None,
    square_feet: Optional[int] = None,
    gentle: bool = False,
    selection_pause: float = 0.3,
    select_retries: int = 2,
) -> Dict[str, Any]:
    driver = build_driver(headless=headless)
    try:
        data: Optional[Dict[str, Any]] = None
        last_error: Optional[str] = None
        for url in CANDIDATE_URLS:
            try:
                data = try_one_url(
                    driver,
                    url,
                    address,
                    timeout,
                    debug=debug,
                    email=email,
                    password=password,
                    max_scrolls=max_scrolls,
                    scroll_pause=scroll_pause,
                    synthesize_range=synthesize_range,
                    range_low_pct=range_low_pct,
                    range_high_pct=range_high_pct,
                    use_filters=use_filters,
                    property_type=property_type,
                    bedrooms=bedrooms,
                    bathrooms=bathrooms,
                    square_feet=square_feet,
                    gentle=gentle,
                    selection_pause=selection_pause,
                    select_retries=select_retries,
                )
                if data:
                    break
            except NoSuchWindowException as e:
                last_error = str(e)
                # Attempt to rebuild driver and continue with next URL
                try:
                    driver.quit()
                except Exception:
                    pass
                driver = build_driver(headless=headless)
                continue
            except Exception as e:
                last_error = str(e)
                continue
        # If nothing parsed and user opted to manually drive the UI, let them do it then capture.
        if not data and wait_for_user:
            if headless:
                print("[INFO] --wait-for-user requires a visible browser; reopening without headless...")
                try:
                    driver.quit()
                except Exception:
                    pass
                driver = build_driver(headless=False)
            url = CANDIDATE_URLS[0]
            print(f"Opening for manual interaction: {url}")
            driver.get(url)
            try:
                _ = input("If needed, sign in and perform the RentCast search in the browser window, then press Enter here to capture results...")
            except Exception:
                pass
            # After user interaction, attempt a parse of current (scrolled) page
            txt = scroll_and_capture_text(driver, max_scrolls=max_scrolls, pause=scroll_pause)
            data = parse_fields_from_text(txt)
            labeled2 = parse_labeled_fields(txt)
            data = merge_prefer_right(data, labeled2)
            try:
                dom_vals2 = parse_dom_fields(driver)
                data = merge_prefer_right(data, dom_vals2)
            except Exception:
                pass
            json_vals2 = parse_json_fields(driver.page_source)
            data = merge_prefer_right(data, json_vals2)
            had_labeled_high2 = False
            try:
                had_labeled_high2 = (
                    (labeled2.get("high_estimate") is not None)
                    or ((locals().get("dom_vals2") or {}).get("high_estimate") is not None)  # type: ignore[attr-defined]
                    or (json_vals2.get("high_estimate") is not None)
                )
                if (
                    isinstance(data.get("low_estimate"), int)
                    and isinstance(data.get("high_estimate"), int)
                    and data["low_estimate"] == data["high_estimate"]
                    and not had_labeled_high2
                    and not re.search(r"(?:rent|range)[^$\n]{0,60}\$\s*[\d,\.]+\s*[-–]\s*\$\s*[\d,\.]+", txt, re.I)
                ):
                    data["high_estimate"] = None
            except Exception:
                pass
            # Synthesize range when requested and still missing a meaningful high
            if synthesize_range and data.get("estimated_rent") is not None:
                has_explicit_range2 = bool(re.search(r"(?:rent|range)[^$\n]{0,60}\$\s*[\d,\.]+\s*[-–]\s*\$\s*[\d,\.]+", txt, re.I))
                if (data.get("high_estimate") is None) or (
                    isinstance(data.get("low_estimate"), int)
                    and isinstance(data.get("high_estimate"), int)
                    and data["low_estimate"] == data["high_estimate"]
                    and not had_labeled_high2
                    and not has_explicit_range2
                ):
                    r2 = float(data["estimated_rent"])  # type: ignore
                    if data.get("low_estimate") is None:
                        data["low_estimate"] = int(round(r2 * range_low_pct))
                    computed_high2 = int(round(r2 * range_high_pct))
                    if data.get("high_estimate") is None:
                        data["high_estimate"] = computed_high2
            if synthesize_range and data.get("estimated_rent") is not None and (data.get("low_estimate") is None and data.get("high_estimate") is None):
                r = float(data["estimated_rent"])  # type: ignore
                data["low_estimate"] = int(round(r * range_low_pct))
                data["high_estimate"] = int(round(r * range_high_pct))
            data = normalize_fields(data)
            if all(data.get(k) is None for k in ["estimated_rent", "low_estimate", "high_estimate", "beds", "baths", "sq_ft"]):
                data = None
        result = {
            "address": address,
            "source": "rentcast_site",
            **(data or {"estimated_rent": None, "low_estimate": None, "high_estimate": None, "beds": None, "baths": None, "sq_ft": None}),
            "note": ("Parsed via Selenium" if data else (f"No fields parsed. Last error: {last_error}" if last_error else "No fields parsed.")),
        }
        # Bubble up filter metadata if present inside data
        if isinstance(data, dict) and 'filter_metadata' in data:
            result['filter_metadata'] = data['filter_metadata']
        save_results(result)
        return result
    finally:
        try:
            driver.quit()
        except Exception:
            pass


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape RentCast site (best-effort) to get rent/beds/baths/sqft using Selenium.")
    p.add_argument("address", help="Property address string")
    p.add_argument("--no-headless", action="store_true", help="Run Chrome with a visible window")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Wait timeout in seconds")
    p.add_argument("--debug", action="store_true", help="Save last page HTML and screenshot for debugging")
    p.add_argument("--wait-for-user", action="store_true", help="Open the site and let you search manually, then capture and parse the page when you press Enter")
    p.add_argument("--scrolls", type=int, default=8, help="How many incremental scrolls to attempt for lazy-loaded content")
    p.add_argument("--scroll-pause", type=float, default=0.8, help="Pause (seconds) between scrolls")
    p.add_argument("--email", help="RentCast login email (optional; used if login is required)")
    p.add_argument("--password", help="RentCast login password (optional; used if login is required)")
    p.add_argument("--synthesize-range", action="store_true", help="If only a single rent value is visible, synthesize low/high using percentages instead of leaving them null")
    p.add_argument("--range-low-pct", type=float, default=0.9, help="Multiplier for estimated_rent to compute low estimate when synthesizing (default 0.9)")
    p.add_argument("--range-high-pct", type=float, default=1.15, help="Multiplier for estimated_rent to compute high estimate when synthesizing (default 1.15)")
    
    # Property filter options
    p.add_argument("--use-filters", action="store_true", help="Enable filling property feature filters (property type, beds, baths, sqft)")
    p.add_argument("--property-type", help="Property type (e.g., 'Single Family', 'Multi-Family', 'Condo', 'Townhouse')")
    p.add_argument("--beds", type=int, help="Number of bedrooms")
    p.add_argument("--baths", type=float, help="Number of bathrooms (can be decimal like 2.5)")
    p.add_argument("--sqft", type=int, help="Square footage")
    p.add_argument("--gentle", action="store_true", help="Use slower, more careful interactions for filter selection")
    p.add_argument("--selection-pause", type=float, default=0.3, help="Pause duration after filter selections in seconds (default 0.3)")
    p.add_argument("--select-retries", type=int, default=2, help="Number of retry attempts for filter selections (default 2)")
    
    return p.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv or sys.argv[1:])
    headless = not args.no_headless
    
    # Show filter info if enabled
    if args.use_filters:
        print("\n" + "="*70)
        print("PROPERTY FILTERS ENABLED")
        print("="*70)
        if args.property_type:
            print(f"  Property Type: {args.property_type}")
        if args.beds is not None:
            print(f"  Bedrooms: {args.beds}")
        if args.baths is not None:
            print(f"  Bathrooms: {args.baths}")
        if args.sqft is not None:
            print(f"  Square Feet: {args.sqft}")
        print("="*70 + "\n")
    
    run(
        args.address,
        headless=headless,
        timeout=args.timeout,
        debug=args.debug,
        wait_for_user=args.wait_for_user,
        email=args.email,
        password=args.password,
        max_scrolls=args.scrolls,
        scroll_pause=args.scroll_pause,
        synthesize_range=args.synthesize_range,
        range_low_pct=args.range_low_pct,
        range_high_pct=args.range_high_pct,
        use_filters=args.use_filters,
        property_type=args.property_type,
        bedrooms=args.beds,
        bathrooms=args.baths,
        square_feet=args.sqft,
        gentle=args.gentle,
        selection_pause=args.selection_pause,
        select_retries=args.select_retries,
    )


if __name__ == "__main__":
    main()
