"""
src/extract.py
--------------
THE EXTRACT LAYER

PURPOSE:
    Connects to the NewsAPI and retrieves the latest financial news articles
    as raw JSON. This module's only job is to GET data and return it unchanged.

ETL CONCEPT — Why keep Extract separate?
    In enterprise pipelines, the Extract layer is intentionally "dumb."
    It does not clean, rename, or interpret data. This means:
      1. If the API changes its response format, only this file needs updating.
      2. We can always re-run the extract and get the same raw data again.
      3. The raw data can be stored as-is for auditing and compliance.

    This is called the "Bronze Layer" in the Medallion Architecture used by
    companies like Databricks, Netflix, and Uber.

WHY top-headlines INSTEAD OF everything?
    NewsAPI's free tier redacts articles older than ~24 hours — it replaces
    title, description, and content with the string '[Removed]'. The
    /v2/everything endpoint searches archives and hits this redaction wall.

    The /v2/top-headlines endpoint always returns LIVE, current headlines
    that are never redacted. This is the correct endpoint for a daily
    pipeline that runs fresh every morning.

USAGE:
    from src.extract import fetch_market_news
    articles = fetch_market_news()
"""

import os
import logging
import requests
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Constants ---
# top-headlines: always live, never redacted — correct for a daily pipeline
NEWS_API_BASE_URL = "https://newsapi.org/v2/top-headlines"

# Category filter — 'business' returns financial and market news directly
NEWS_CATEGORY = "business"

# Country — 'us' gives US market news (change to 'gb' for UK, etc.)
NEWS_COUNTRY = "us"

# Max articles per run.
# 5 = safe for testing. Each run uses 5 Groq API calls out of 14,400 daily free tier.
# Increase to 10-20 once pipeline is stable in production.
MAX_ARTICLES = 5


def fetch_market_news() -> list[dict]:
    """
    Fetches the latest live financial headlines from NewsAPI.

    Uses /v2/top-headlines with category=business to get current,
    never-redacted articles. Filters out any '[Removed]' articles
    defensively before returning.

    Returns:
        A list of raw article dicts. Each has: title, description,
        source, author, url, publishedAt.

    Raises:
        ValueError: If NEWS_API_KEY is not set.
        requests.HTTPError: If the API returns a non-200 response.
    """
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        raise ValueError("NEWS_API_KEY is not set in your .env file.")

    params = {
        "category": NEWS_CATEGORY,
        "country":  NEWS_COUNTRY,
        "pageSize": MAX_ARTICLES,
        "apiKey":   api_key,
    }

    logger.info(f"Fetching top business headlines from NewsAPI...")

    try:
        response = requests.get(NEWS_API_BASE_URL, params=params, timeout=10)
        response.raise_for_status()

    except requests.exceptions.Timeout:
        logger.error("NewsAPI request timed out after 10 seconds.")
        raise
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to NewsAPI. Check your internet connection.")
        raise
    except requests.exceptions.HTTPError as e:
        logger.error(f"NewsAPI returned an error: {e}")
        raise

    data = response.json()

    if data.get("status") != "ok":
        error_msg = data.get("message", "Unknown error from NewsAPI.")
        logger.error(f"NewsAPI error: {error_msg}")
        raise RuntimeError(f"NewsAPI error: {error_msg}")

    articles = data.get("articles", [])

    # Defensive filter — remove any redacted articles before they reach Transform
    clean = [
        a for a in articles
        if a.get("title") and a.get("title") != "[Removed]"
    ]

    removed_count = len(articles) - len(clean)
    if removed_count:
        logger.warning(f"Filtered out {removed_count} redacted '[Removed]' articles.")

    logger.info(f"Successfully extracted {len(clean)} live articles.")
    return clean


# ---------------------------------------------------------------------------
# Quick test
# Usage: python -m src.extract
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("=== Running Extract Layer Test ===")
    articles = fetch_market_news()

    if not articles:
        logger.warning("No articles returned.")
    else:
        for a in articles:
            print(f"  [{a.get('source', {}).get('name')}]  {a.get('title')}")
        print(f"\n✅  {len(articles)} live articles ready for Transform.\n")
