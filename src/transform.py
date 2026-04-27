"""
src/transform.py
----------------
THE TRANSFORM LAYER

PURPOSE:
    Takes raw article dictionaries from the Extract layer and enriches each
    one with AI-generated market intelligence using the Groq API.
    Returns a list of structured, analytics-ready records.

ETL CONCEPT — Why is Transform the most important layer?
    Raw data is useless without meaning. A news headline like
    "Fed raises rates by 50bps" is just text. The Transform layer's job is
    to convert that text into FACTS a database can store and a BI tool can query:
        sentiment  = "Bearish"
        entity     = "Federal Reserve"
        score      = 0.82

    This is the "Silver Layer" in the Medallion Architecture — cleaned and
    enriched data that sits between raw (Bronze) and final reporting (Gold).

WHY GROQ?
    Groq runs open-source Llama models on custom LPU hardware. The free tier
    gives 14,400 requests/day and 30 requests/minute — enough for our pipeline
    to process 10 articles in ~20 seconds with zero rate limit issues.
    No billing card required.

USAGE:
    from src.transform import transform_articles
    enriched = transform_articles(raw_articles)
"""

import os
import json
import time
import logging
import re
import random
from groq import Groq
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# --- Logger ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Constants ---
# llama-3.1-8b-instant: Fast, accurate, handles structured JSON output perfectly.
# Free tier: 14,400 req/day, 30 RPM — no rate limit issues for our pipeline.
GROQ_MODEL = "llama-3.1-8b-instant"

# Delay between API calls — Groq allows 30 RPM, so 2 seconds is plenty of buffer
API_CALL_DELAY_SECONDS = 2

# Max retries on rate limit
MAX_RETRIES = 3

# Max characters of description to send — keeps token usage lean
MAX_DESCRIPTION_CHARS = 200

# ---------------------------------------------------------------------------
# MOCK MODE — controlled by GEMINI_MOCK_MODE in .env
# When true, bypasses all AI calls entirely. Useful for testing the
# Load layer and orchestration without consuming any API quota.
# ---------------------------------------------------------------------------
MOCK_MODE = os.getenv("GEMINI_MOCK_MODE", "false").lower() == "true"

MOCK_RESPONSES = [
    {"sentiment": "Bearish",  "sentiment_score": 0.78, "key_entity": "Federal Reserve",    "key_entity_type": "Economic Indicator", "one_line_summary": "Fed rate uncertainty weighs on equity markets amid inflation concerns."},
    {"sentiment": "Bullish",  "sentiment_score": 0.82, "key_entity": "Apple Inc.",          "key_entity_type": "Company",            "one_line_summary": "Strong iPhone demand drives Apple earnings beat expectations."},
    {"sentiment": "Neutral",  "sentiment_score": 0.55, "key_entity": "Crude Oil",           "key_entity_type": "Commodity",          "one_line_summary": "Oil prices stabilize as supply and demand signals remain mixed."},
    {"sentiment": "Bearish",  "sentiment_score": 0.71, "key_entity": "U.S. Treasury Bonds", "key_entity_type": "Economic Indicator", "one_line_summary": "Rising yields signal bond market stress as investors reassess risk."},
    {"sentiment": "Bullish",  "sentiment_score": 0.88, "key_entity": "NVIDIA Corporation",  "key_entity_type": "Company",            "one_line_summary": "AI chip demand surge positions NVIDIA for record revenue quarter."},
    {"sentiment": "Neutral",  "sentiment_score": 0.50, "key_entity": "S&P 500",             "key_entity_type": "Economic Indicator", "one_line_summary": "Markets trade sideways as investors await key economic data releases."},
    {"sentiment": "Bearish",  "sentiment_score": 0.65, "key_entity": "Supply Chain",        "key_entity_type": "Sector",             "one_line_summary": "Global supply disruptions raise costs and pressure corporate margins."},
    {"sentiment": "Bullish",  "sentiment_score": 0.74, "key_entity": "Jerome Powell",       "key_entity_type": "Person",             "one_line_summary": "Fed chair signals potential rate cuts boosting investor confidence."},
]

# ---------------------------------------------------------------------------
# PROMPT TEMPLATE — Token-Optimized
# ---------------------------------------------------------------------------
# Compact prompt = fewer tokens = faster calls = more free tier headroom.
# The JSON schema itself is the instruction — no prose padding needed.
# ---------------------------------------------------------------------------
ANALYSIS_PROMPT_TEMPLATE = """Analyze this financial news. Return ONLY valid JSON, no markdown.
Schema: {{"sentiment":"Bullish|Bearish|Neutral","sentiment_score":0.0-1.0,"key_entity":"<main entity>","key_entity_type":"Company|Person|Economic Indicator|Sector|Commodity","one_line_summary":"<max 15 words>"}}
Title: {title}
Info: {description}"""


def _mock_analyze(article: dict) -> dict:
    """Returns realistic mock AI analysis without any API call."""
    title = (article.get("title") or "").strip()
    description = (article.get("description") or "").strip() or "No description provided."
    mock = random.choice(MOCK_RESPONSES)
    return {
        "title":            title,
        "description":      description,
        "url":              article.get("url", ""),
        "source_name":      article.get("source", {}).get("name", "Unknown"),
        "author":           article.get("author", "Unknown"),
        "published_at":     article.get("publishedAt", ""),
        "sentiment":        mock["sentiment"],
        "sentiment_score":  mock["sentiment_score"],
        "key_entity":       mock["key_entity"],
        "key_entity_type":  mock["key_entity_type"],
        "one_line_summary": mock["one_line_summary"],
    }


def _configure_groq() -> Groq:
    """Initializes and returns the Groq client."""
    api_key = os.getenv("GROQ_API_KEY", "").strip().strip("'\"")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com"
        )
    logger.info(f"Groq key loaded: {api_key[:8]}...{api_key[-4:]} | Model: {GROQ_MODEL}")
    client = Groq(api_key=api_key)
    return client


def _call_groq_with_retry(client: Groq, prompt: str) -> str:
    """
    Calls Groq with retry on rate limit errors.

    Groq's free tier is generous (30 RPM) so this should rarely trigger.
    When it does, we read the suggested retry delay from the error response
    and wait exactly that long before retrying.
    """
    fallback_wait = 10

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,   # Low temperature = more consistent JSON output
                max_tokens=200,    # Our JSON response is never more than ~100 tokens
            )
            return completion.choices[0].message.content

        except Exception as e:
            error_str = str(e)
            is_rate_limit = (
                "429" in error_str
                or "rate_limit" in error_str.lower()
                or "quota" in error_str.lower()
            )

            if is_rate_limit and attempt < MAX_RETRIES:
                match = re.search(r"retry after (\d+(?:\.\d+)?)", error_str, re.IGNORECASE)
                wait = float(match.group(1)) + 2 if match else fallback_wait
                logger.warning(
                    f"  Rate limited (attempt {attempt}/{MAX_RETRIES}). "
                    f"Waiting {wait:.0f}s..."
                )
                time.sleep(wait)
            else:
                raise


def _parse_response(raw_text: str) -> dict:
    """Parses the AI response into a Python dict, stripping any markdown wrappers."""
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw_text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"AI returned invalid JSON.\nRaw: {raw_text}\nError: {e}")


def _validate_analysis(analysis: dict) -> dict:
    """Enforces schema — fills safe defaults for any missing or invalid fields."""
    valid_sentiments = {"Bullish", "Bearish", "Neutral"}

    if analysis.get("sentiment") not in valid_sentiments:
        analysis["sentiment"] = "Neutral"

    if not isinstance(analysis.get("sentiment_score"), (int, float)):
        analysis["sentiment_score"] = 0.5

    analysis["sentiment_score"] = max(0.0, min(1.0, float(analysis["sentiment_score"])))
    analysis.setdefault("key_entity", "Unknown")
    analysis.setdefault("key_entity_type", "Unknown")
    analysis.setdefault("one_line_summary", "No summary available.")
    return analysis


def _analyze_single_article(client: Groq, article: dict) -> dict | None:
    """Sends one article to Groq and returns the enriched record."""
    # Use `or ""` instead of `.get(key, "")` — the default only applies when
    # the key is MISSING. NewsAPI sets null values explicitly (e.g. "description": null),
    # which becomes None in Python. `or ""` handles both missing AND null safely.
    title = (article.get("title") or "").strip()
    description = (article.get("description") or "").strip() or "No description."

    if not title or title == "[Removed]":
        logger.warning(f"Skipping article — title is empty or removed. Raw title: '{title}'")
        return None

    description = description[:MAX_DESCRIPTION_CHARS]
    prompt = ANALYSIS_PROMPT_TEMPLATE.format(title=title, description=description)

    try:
        raw_text = _call_groq_with_retry(client, prompt)
        analysis = _parse_response(raw_text)
        analysis = _validate_analysis(analysis)

        return {
            "title":            title,
            "description":      description,
            "url":              article.get("url", ""),
            "source_name":      article.get("source", {}).get("name", "Unknown"),
            "author":           article.get("author", "Unknown"),
            "published_at":     article.get("publishedAt", ""),
            "sentiment":        analysis["sentiment"],
            "sentiment_score":  analysis["sentiment_score"],
            "key_entity":       analysis["key_entity"],
            "key_entity_type":  analysis["key_entity_type"],
            "one_line_summary": analysis["one_line_summary"],
        }

    except Exception as e:
        # Poison pill handling — log and skip, don't crash the pipeline
        logger.error(f"Failed to analyze '{title[:60]}': {type(e).__name__}: {e}")
        return None


def transform_articles(raw_articles: list[dict]) -> list[dict]:
    """
    Main Transform function. Enriches raw articles with AI sentiment analysis.

    Args:
        raw_articles: List of article dicts from fetch_market_news().

    Returns:
        List of enriched dicts ready for the Load layer.
    """
    if not raw_articles:
        logger.warning("No articles received for transformation.")
        return []

    enriched_articles = []

    # --- Mock mode: bypass all AI calls ---
    if MOCK_MODE:
        logger.warning(
            "⚠️  MOCK_MODE=true — using synthetic AI responses. "
            "Set GEMINI_MOCK_MODE=false in .env to use real Groq API."
        )
        for article in raw_articles:
            title = article.get("title", "").strip()
            if title not in ("", "[Removed]"):
                enriched_articles.append(_mock_analyze(article))
            else:
                logger.warning(f"Mock mode skipping article — title: '{title}'")
        logger.info(f"Mock transform complete. {len(enriched_articles)} records ready.")
        return enriched_articles

    # --- Real mode: call Groq API ---
    client = _configure_groq()
    logger.info(f"Starting transformation of {len(raw_articles)} articles via Groq...")

    for i, article in enumerate(raw_articles, start=1):
        logger.info(
            f"  Analyzing article {i}/{len(raw_articles)}: "
            f"{article.get('title', '')[:60]}..."
        )

        result = _analyze_single_article(client, article)
        if result:
            enriched_articles.append(result)

        if i < len(raw_articles):
            time.sleep(API_CALL_DELAY_SECONDS)

    logger.info(
        f"Transform complete. "
        f"{len(enriched_articles)} enriched / "
        f"{len(raw_articles) - len(enriched_articles)} skipped."
    )
    return enriched_articles


# ---------------------------------------------------------------------------
# Quick test
# Usage: python -m src.transform
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.extract import fetch_market_news

    logger.info("=== Running Transform Layer Test ===")
    raw = fetch_market_news()

    if not raw:
        logger.error("No articles fetched.")
    else:
        test_sample = raw[:2]
        logger.info(f"Transforming {len(test_sample)} articles (sample)...")
        enriched = transform_articles(test_sample)

        print("\n" + "=" * 60)
        print("  TRANSFORMED RECORDS")
        print("=" * 60)
        for record in enriched:
            print(f"\n  Title      : {record['title'][:70]}")
            print(f"  Sentiment  : {record['sentiment']}  (score: {record['sentiment_score']})")
            print(f"  Key Entity : {record['key_entity']}  [{record['key_entity_type']}]")
            print(f"  Summary    : {record['one_line_summary']}")
            print(f"  Source     : {record['source_name']}  |  {record['published_at']}")
        print("\n" + "=" * 60)
        print(f"  ✅  Transform layer working. {len(enriched)} records ready for Load.\n")
