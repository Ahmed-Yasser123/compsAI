import asyncio
import sys
from scraper.stub import fetch_property_data

addr = " ".join(sys.argv[1:]) or "1600 Pennsylvania Ave NW, Washington, DC"
res = asyncio.run(fetch_property_data(address=addr))
print("Scraper returned:", res)
print("Output JSON:", res.get("scrapeOutputPath"))