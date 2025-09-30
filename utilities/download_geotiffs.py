"""Download NSIDC sea ice extent GeoTIFF files with robust retry logic.

This module provides utilities for downloading daily sea ice GeoTIFF files from
NSIDC's G02135 dataset. It implements exponential backoff, rate limiting, and
proper HTTP error handling for reliable bulk downloads.
"""

import email.utils as eut
import os
import random
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

RETRYABLE_STATUS: set[int] = {500, 502, 503, 504}

def _parse_retry_after(value: str) -> Optional[float]:
    """Parse the Retry-After HTTP header value into seconds.

    Args:
        value: Retry-After header value (either seconds or HTTP-date format).

    Returns:
        Delay in seconds if parseable, None otherwise.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            dt = eut.parsedate_to_datetime(value)
            return max(0.0, (dt - dt.now(dt.tzinfo)).total_seconds())
        except Exception:
            return None


def _sleep_with_backoff(
    attempt: int,
    base_delay: float,
    cap: float,
    retry_after: Optional[float]
) -> None:
    """Sleep with exponential backoff or use server-provided retry delay.

    Implements jittered exponential backoff: delay = random(0, min(cap, base * 2^(attempt-1)))
    If server provides Retry-After header, use that instead (capped at max).

    Args:
        attempt: Current attempt number (1-indexed).
        base_delay: Base delay in seconds for exponential backoff.
        cap: Maximum delay cap in seconds.
        retry_after: Server-specified retry delay in seconds, if provided.
    """
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
    """Make HTTP request with automatic retries for transient failures.

    Retries on connection errors, timeouts, and 5xx server errors using
    exponential backoff with jitter. Respects Retry-After headers.

    Args:
        session: Requests session to use for the request.
        method: HTTP method (GET, POST, etc.).
        url: Target URL.
        max_attempts: Maximum number of attempts before giving up.
        base_delay: Base delay for exponential backoff in seconds.
        cap_delay: Maximum delay cap in seconds.
        timeout: Connection and read timeout tuple (connect, read).
        **kwargs: Additional arguments passed to session.request().

    Returns:
        Successful HTTP response.

    Raises:
        requests.HTTPError: If max attempts reached with server error.
        requests.ConnectionError: If max attempts reached with connection error.
        requests.Timeout: If max attempts reached with timeout.
    """
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
    """Create a requests session with appropriate User-Agent header.

    Returns:
        Configured requests session for NSIDC data access.
    """
    s = requests.Session()
    s.headers.update({"User-Agent": "SMF-data-fetch/1.0 (+contact: o.stein@smf.de)"})
    return s


BASE_URL: str = "https://noaadata.apps.nsidc.org/NOAA/G02135/north/daily/geotiff/"


def list_links(
    session: requests.Session,
    url: str,
    suffix: Optional[str] = None
) -> list[str]:
    """Scrape and return all links from an HTML directory listing.

    Args:
        session: Requests session to use.
        url: URL of the directory page to scrape.
        suffix: Optional file extension filter (e.g., '.tif').

    Returns:
        List of absolute URLs found on the page.
    """
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


def download_file(session: requests.Session, url: str, dest_path: str) -> None:
    """Download a file from URL to local path with streaming.

    Creates parent directories if they don't exist.

    Args:
        session: Requests session to use.
        url: URL of the file to download.
        dest_path: Local filesystem path for the downloaded file.
    """
    with request_with_retries(session, "GET", url, stream=True) as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=62914560):
                if chunk:
                    f.write(chunk)


def download_all_daily_geotiffs(output_dir: str) -> None:
    """Download all available daily GeoTIFF files from NSIDC G02135.

    Crawls the NSIDC directory structure (year/month/files) and downloads
    all .tif files. Skips files that already exist locally. Implements
    rate limiting with 0.2s delay between downloads.

    Args:
        output_dir: Root directory for downloaded files (organized as year/month/file.tif).
    """
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
