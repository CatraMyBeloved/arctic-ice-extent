import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

import time, random, email.utils as eut
import requests

RETRYABLE_STATUS = {500, 502, 503, 504}

def _parse_retry_after(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            dt = eut.parsedate_to_datetime(value)
            return max(0.0, (dt - dt.now(dt.tzinfo)).total_seconds())
        except Exception:
            return None

def _sleep_with_backoff(attempt: int, base_delay: float, cap: float, retry_after: float | None):
    if retry_after is not None:
        delay = min(retry_after, cap)
    else:
        delay = min(cap, base_delay * (2 ** (attempt - 1)))
        delay = random.uniform(0, delay)
    time.sleep(delay)

def request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    *,
    max_attempts: int = 5,
    base_delay: float = 2.0,
    cap_delay: float = 120.0,
    timeout: tuple[float, float] = (10.0, 60.0),
    **kwargs
) -> requests.Response:

    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.request(method, url, timeout=timeout, **kwargs)
            if resp.status_code in RETRYABLE_STATUS:
                if attempt == max_attempts:
                    resp.raise_for_status()
                retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                _sleep_with_backoff(attempt, base_delay, cap_delay, retry_after)
                continue
            return resp
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            if attempt == max_attempts:
                raise
            _sleep_with_backoff(attempt, base_delay, cap_delay, None)
    if last_exc:
        raise last_exc


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "SMF-data-fetch/1.0 (+contact: o.stein@smf.de)"})
    return s

BASE_URL = "https://noaadata.apps.nsidc.org/NOAA/G02135/north/daily/geotiff/"

def list_links(session, url, suffix=None):
    resp = request_with_retries(session, "GET", url)
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


def download_file(session, url, dest_path):
    with request_with_retries(session, "GET", url, stream=True) as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=62914560):
                if chunk:
                    f.write(chunk)


def download_all_daily_geotiffs(output_dir):
    session = make_session()

    year_urls = [u for u in list_links(session, BASE_URL) if u.rstrip("/").split("/")[-1].isdigit()]
    for year_url in year_urls:
        year = year_url.rstrip("/").split("/")[-1]
        print(f"Processing year {year}")
        month_urls = list_links(session, year_url)
        for month_url in month_urls:
            month = month_url.rstrip("/").split("/")[-1]
            print(f"  Month {month}")
            tif_urls = [u for u in list_links(session, month_url, suffix=".tif")]
            for tif_url in tif_urls:
                filename = tif_url.split("/")[-1]
                dest = os.path.join(output_dir, year, month, filename)
                if not os.path.exists(dest):
                    print(f"    Downloading {filename}")
                    download_file(session, tif_url, dest)
                    time.sleep(0.2)
                else:
                    print(f"    Skipping existing {filename}")





if __name__ == "__main__":
    output_dir = "../data/raw/geotiffs/daily/"
    download_all_daily_geotiffs(output_dir)
