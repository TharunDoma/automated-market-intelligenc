"""
main.py
-------
PIPELINE ORCHESTRATOR

PURPOSE:
    Single entry point that chains all three ETL layers in sequence.
    This is the file GitHub Actions will call every morning at 8 AM.

USAGE:
    python main.py

ETL CONCEPT — Why a separate orchestrator?
    Each layer (extract, transform, load) is independently testable.
    main.py is the "conductor" — it calls each layer in order, passes
    data between them, and produces a final run summary. If any layer
    fails, the exception propagates here and GitHub Actions marks the
    run as failed, triggering an alert email.
"""

import logging
import sys
from datetime import datetime

from src.extract import fetch_market_news
from src.transform import transform_articles
from src.load import load_to_snowflake

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_pipeline() -> dict:
    """
    Executes the full ETL pipeline and returns a summary dict.
    Raises on any unrecoverable error so GitHub Actions marks the run failed.
    """
    start_time = datetime.utcnow()
    logger.info("=" * 55)
    logger.info("  MARKET INTELLIGENCE PIPELINE — STARTING")
    logger.info(f"  Run started at: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    logger.info("=" * 55)

    # --- EXTRACT ---
    logger.info("▶  Step 1/3: Extract")
    raw_articles = fetch_market_news()
    if not raw_articles:
        logger.warning("No articles extracted. Exiting early.")
        return {"extracted": 0, "transformed": 0, "loaded": 0}

    # --- TRANSFORM ---
    logger.info("▶  Step 2/3: Transform")
    enriched_articles = transform_articles(raw_articles)
    if not enriched_articles:
        logger.warning("No articles transformed. Skipping load.")
        return {"extracted": len(raw_articles), "transformed": 0, "loaded": 0}

    # --- LOAD ---
    logger.info("▶  Step 3/3: Load")
    inserted_count = load_to_snowflake(enriched_articles)

    end_time = datetime.utcnow()
    duration = (end_time - start_time).seconds

    summary = {
        "extracted":   len(raw_articles),
        "transformed": len(enriched_articles),
        "loaded":      inserted_count,
        "skipped":     len(enriched_articles) - inserted_count,
        "duration_s":  duration,
    }

    logger.info("=" * 55)
    logger.info("  PIPELINE RUN COMPLETE")
    logger.info("=" * 55)
    logger.info(f"  Articles extracted  : {summary['extracted']}")
    logger.info(f"  Articles transformed: {summary['transformed']}")
    logger.info(f"  Records inserted    : {summary['loaded']}")
    logger.info(f"  Records skipped     : {summary['skipped']} (duplicates)")
    logger.info(f"  Total duration      : {summary['duration_s']}s")
    logger.info("=" * 55)

    return summary


if __name__ == "__main__":
    try:
        summary = run_pipeline()
        # Exit 0 = success → GitHub Actions marks run as green
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed with unhandled error: {e}", exc_info=True)
        # Exit 1 = failure → GitHub Actions marks run as red and sends alert email
        sys.exit(1)
