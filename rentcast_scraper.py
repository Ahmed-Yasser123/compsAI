
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# requests is optional when using stub mode (no API calls)
try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore

API_BASE = "https://api.rentcast.io/v1/estimates/rent"
OUTPUT_FILENAME = "rentcast_results.json"


def _stub_estimate(address: str) -> Dict[str, Any]:
    """Deterministic fallback when no API key is provided or API request fails.

    Uses a hash of the address to produce plausible-looking values.
    """
    import hashlib

    h = int(hashlib.sha1(address.encode("utf-8")).hexdigest()[:8], 16)
    # Beds 1–5, Baths 1.0–3.5, Sqft 800–3200, rent ~900–3500
    beds = 1 + (h % 5)
    baths = 1.0 + ((h >> 3) % 6) * 0.5
    sqft = 800 + ((h >> 5) % 2401)  # up to ~3200
    base = 900 + ((h >> 7) % 2601)  # 900–3500
    low = int(base * 0.9)
    high = int(base * 1.15)
    est = int((low + high) / 2)
    return {
        "address": address,
        "beds": beds,
        "baths": round(baths, 1),
        "sq_ft": sqft,
        "estimated_rent": est,
        "low_estimate": low,
        "high_estimate": high,
    }


def fetch_rentcast(address: str, api_key: str | None) -> Dict[str, Any] | None:
    """Fetch rent estimate data via API; if api_key is falsy or requests missing, return None to indicate fallback."""
    if not api_key:
        return None
    if requests is None:
        print("[WARN] 'requests' not installed; cannot call API. Falling back to stub.")
        return None
    headers = {"X-Api-Key": api_key.strip(), "Accept": "application/json"}
    try:
        resp = requests.get(API_BASE, params={"address": address}, headers=headers, timeout=15)
    except Exception as e:
        print(f"[WARN] Network error for address '{address}': {e}")
        return None

    if resp.status_code != 200:
        print(f"[WARN] Non-200 ({resp.status_code}) for address '{address}'. Body: {resp.text[:200]}")
        return None

    try:
        data = resp.json()
    except Exception as e:
        print(f"[WARN] JSON parse error for '{address}': {e}")
        return None

    # Map the required fields (gracefully handle missing keys)
    out = {
        "address": address,
        "beds": data.get("bedrooms"),
        "baths": data.get("bathrooms"),
        "sq_ft": data.get("squareFootage"),
        "estimated_rent": data.get("rent"),
        "low_estimate": data.get("rentLow"),
        "high_estimate": data.get("rentHigh"),
    }
    return out


def normalize_addresses(raw: str) -> List[str]:
    """Split comma/newline separated addresses; strip empties."""
    parts: List[str] = []
    for chunk in raw.replace("\r", "").split("\n"):
        parts.extend([p.strip() for p in chunk.split(",")])
    return [p for p in parts if p]


def save_results(results: List[Dict[str, Any]]):
    path = Path(__file__).resolve().parent / OUTPUT_FILENAME
    if len(results) == 1:
        payload: Any = results[0]
    else:
        payload = results
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} record(s) to {path}")


def interactive_flow():
    api_key = input("Enter RentCast API key (optional, press Enter to skip for stub mode): ").strip()
    addr_input = input("Enter property address (or multiple separated by commas/newlines): ").strip()
    addresses = normalize_addresses(addr_input)
    if not addresses:
        print("At least one address is required.")
        sys.exit(1)

    # In stub mode, offer rent-related overrides for each address.
    manual_overrides: Dict[str, Dict[str, Any]] = {}
    if not api_key:
        print("\nStub mode: You can optionally enter overrides. Leave blank to keep generated stub.")
        for addr in addresses:
            print(f"\nOverrides for: {addr}")
            beds_in = input("  Beds (int): ").strip()
            baths_in = input("  Baths (float): ").strip()
            sqft_in = input("  Sq Ft (int): ").strip()
            rent_in = input("  Est. Rent (int): ").strip()
            low_in = input("  Low Est. (int): ").strip()
            high_in = input("  High Est. (int): ").strip()
            ov: Dict[str, Any] = {}
            if beds_in: ov["beds"] = int(beds_in)
            if baths_in: ov["baths"] = float(baths_in)
            if sqft_in: ov["sq_ft"] = int(sqft_in)
            if rent_in: ov["estimated_rent"] = int(rent_in)
            if low_in: ov["low_estimate"] = int(low_in)
            if high_in: ov["high_estimate"] = int(high_in)
            if ov:
                manual_overrides[addr] = ov

    # Collect property details for all modes (API or stub)
    print("\nProperty details (units/buildings/HOA/taxes/insurance). Leave blank to skip per field.")
    for addr in addresses:
        print(f"\nDetails for: {addr}")
        units_in = input("  Units (#): ").strip()
        buildings_in = input("  Buildings (#): ").strip()
        hoa_in = input("  HOA fee (e.g., 0 if none): ").strip()
        taxes_in = input("  Taxes per year: ").strip()

        # Determine insurance per rule: if units >= 5 ask, else default 1500; allow explicit override
        insurance_val: Any = None
        units_val: Any = int(units_in) if units_in else None
        if units_val is not None and units_val < 5:
            insurance_val = 1500
            print("    Insurance auto-set to 1500 (units < 5).")
            ins_override = input("  (Optional) Override insurance: ").strip()
            if ins_override:
                try:
                    insurance_val = float(ins_override)
                except Exception:
                    pass
        else:
            ins_in = input("  Insurance per year (required if units >= 5; optional otherwise): ").strip()
            if ins_in:
                try:
                    insurance_val = float(ins_in)
                except Exception:
                    insurance_val = None

        ov2 = manual_overrides.get(addr, {}).copy()
        if units_in:
            try:
                ov2["units"] = int(units_in)
            except Exception:
                pass
        if buildings_in:
            try:
                ov2["buildings"] = int(buildings_in)
            except Exception:
                pass
        if hoa_in:
            try:
                ov2["hoa_fee"] = float(hoa_in)
            except Exception:
                pass
        if taxes_in:
            try:
                ov2["taxes"] = float(taxes_in)
            except Exception:
                pass
        if insurance_val is not None:
            ov2["insurance"] = insurance_val
        manual_overrides[addr] = ov2

    run(api_key=api_key or None, addresses=addresses, overrides=manual_overrides)


def run(*, api_key: str | None, addresses: List[str], overrides: Dict[str, Dict[str, Any]] | None = None):
    results: List[Dict[str, Any]] = []
    for i, addr in enumerate(addresses, 1):
        print(f"[{i}/{len(addresses)}] Fetching '{addr}'...")
        rec = fetch_rentcast(addr, api_key)
        if rec is None:
            # fallback to stub
            rec = _stub_estimate(addr)
            print("    Using stub estimate (no API).")
        # Apply manual overrides if provided
        if overrides and addr in overrides:
            for k, v in overrides[addr].items():
                rec[k] = v
            print("    Applied manual overrides.")
        results.append(rec)
    if not results:
        print("No successful results to save.")
        return
    save_results(results)
    print("Success: Data retrieval complete.")


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch RentCast rent estimates for one or more addresses.")
    p.add_argument("addresses", nargs="*", help="Address strings (one or many). If omitted, interactive mode is used.")
    p.add_argument("--api-key", dest="api_key", help="RentCast API key. If omitted, interactive prompt is used.")
    # Additional property details (applied to all addresses when provided)
    p.add_argument("--units", dest="units", type=int, help="# of units")
    p.add_argument("--buildings", dest="buildings", type=int, help="# of buildings")
    p.add_argument("--hoa-fee", dest="hoa_fee", type=float, help="HOA fee (0 if none)")
    p.add_argument("--taxes", dest="taxes", type=float, help="Taxes per year")
    p.add_argument("--insurance", dest="insurance", type=float, help="Insurance per year (if units < 5 and not provided, defaults to 1500)")
    return p.parse_args(argv)


def main(argv: List[str] | None = None):
    args = parse_args(argv or sys.argv[1:])
    # If neither addresses nor api key were provided, go interactive
    if not args.addresses and args.api_key is None:
        interactive_flow()
        return
    api_key = (args.api_key.strip() if args.api_key else None)
    addresses = [a.strip() for a in (args.addresses or []) if a.strip()]
    if not addresses:
        # addresses missing -> interactive to collect them (api key may be present or not)
        interactive_flow()
        return
    # Build common overrides from CLI flags and apply to all addresses
    common_ov: Dict[str, Any] = {}
    if getattr(args, "units", None) is not None:
        common_ov["units"] = int(args.units)
    if getattr(args, "buildings", None) is not None:
        common_ov["buildings"] = int(args.buildings)
    if getattr(args, "hoa_fee", None) is not None:
        common_ov["hoa_fee"] = float(args.hoa_fee)
    if getattr(args, "taxes", None) is not None:
        common_ov["taxes"] = float(args.taxes)
    if getattr(args, "insurance", None) is not None:
        common_ov["insurance"] = float(args.insurance)

    # Apply insurance rule if units provided but insurance not provided
    if "units" in common_ov and "insurance" not in common_ov:
        try:
            if int(common_ov["units"]) < 5:
                common_ov["insurance"] = 1500.0
                print("[INFO] Insurance auto-set to 1500 (units < 5). Use --insurance to override.")
            else:
                print("[WARN] Units >= 5 and no --insurance provided. Consider passing --insurance for accuracy.")
        except Exception:
            pass

    overrides: Dict[str, Dict[str, Any]] | None = None
    if common_ov:
        overrides = {addr: common_ov.copy() for addr in addresses}

    run(api_key=api_key, addresses=addresses, overrides=overrides)


if __name__ == "__main__":
    main()
