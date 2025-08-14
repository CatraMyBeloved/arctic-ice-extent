import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://noaadata.apps.nsidc.org/NOAA/G02135/north/daily/geotiff/"

def list_links(url, suffix=None):
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.find_all("a"):
        href = a.get("href")
        if href and href != "../":
            full_url = urljoin(url, href)
            if suffix is None or full_url.endswith(suffix):
                links.append(full_url)
    return links

def download_file(url, dest_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def download_all_daily_geotiffs(output_dir):
    year_urls = [u for u in list_links(BASE_URL) if u.rstrip("/").split("/")[-1].isdigit()]
    for year_url in year_urls:
        year = year_url.rstrip("/").split("/")[-1]
        print(f"Processing year {year}")
        month_urls = list_links(year_url)
        for month_url in month_urls:
            month = month_url.rstrip("/").split("/")[-1]
            print(f"  Month {month}")
            tif_urls = [u for u in list_links(month_url, suffix=".tif")]
            for tif_url in tif_urls:
                filename = tif_url.split("/")[-1]
                dest = os.path.join(output_dir, year, month, filename)
                if not os.path.exists(dest):
                    print(f"    Downloading {filename}")
                    download_file(tif_url, dest)
                else:
                    print(f"    Skipping existing {filename}")

if __name__ == "__main__":
    output_dir = "../data/raw/geotiffs/daily/"
    download_all_daily_geotiffs(output_dir)
