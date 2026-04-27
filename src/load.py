"""
src/load.py
-----------
THE LOAD LAYER

PURPOSE:
    Takes the list of enriched article records from the Transform layer and
    inserts them into Snowflake. Handles table creation, duplicate prevention,
    and safe connection cleanup.

ETL CONCEPT — Why is the Load layer written last?
    The Load layer is the "landing zone" for all the work upstream. It has
    one job: take clean, validated, schema-enforced data and persist it.
    Because the Transform layer already guaranteed the schema, the Load layer
    can be simple and mechanical — no business logic lives here.

    This is the "Gold Layer" in the Medallion Architecture — the final,
    queryable destination that BI tools and analysts will hit directly.

IDEMPOTENCY CONCEPT:
    An idempotent pipeline can be run multiple times and produce the same
    result. We achieve this by skipping articles whose URL already exists
    in the table. Running the pipeline twice on the same day won't create
    duplicate rows — a critical property for any scheduled automation.

USAGE:
    from src.load import load_to_snowflake
    inserted_count = load_to_snowflake(enriched_records)
"""

import os
import logging
import snowflake.connector
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

# ---------------------------------------------------------------------------
# TABLE DEFINITION
# ---------------------------------------------------------------------------
# We define the CREATE TABLE statement here as a constant.
# IF NOT EXISTS = idempotent setup. Running this 100 times has the same
# result as running it once — the table is created only on the first run.
#
# INGESTED_AT: Automatically stamped by Snowflake when the row is inserted.
# This is your audit column — it tells you WHEN the pipeline ran, separate
# from PUBLISHED_AT which is when the article was originally published.
# ---------------------------------------------------------------------------
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS MARKET_NEWS (
    ID              NUMBER AUTOINCREMENT PRIMARY KEY,
    TITLE           VARCHAR(500),
    DESCRIPTION     TEXT,
    URL             VARCHAR(2000),
    SOURCE_NAME     VARCHAR(200),
    AUTHOR          VARCHAR(200),
    PUBLISHED_AT    VARCHAR(50),
    SENTIMENT       VARCHAR(10),
    SENTIMENT_SCORE FLOAT,
    KEY_ENTITY      VARCHAR(300),
    KEY_ENTITY_TYPE VARCHAR(100),
    ONE_LINE_SUMMARY VARCHAR(500),
    INGESTED_AT     TIMESTAMP_TZ DEFAULT CURRENT_TIMESTAMP()
)
"""

# ---------------------------------------------------------------------------
# INSERT STATEMENT (Parameterized)
# ---------------------------------------------------------------------------
# We use %s placeholders and pass values as a tuple — NEVER use f-strings
# or string concatenation to build SQL with user data. That opens you up
# to SQL injection attacks. Parameterized queries are the only safe approach.
# ---------------------------------------------------------------------------
INSERT_SQL = """
INSERT INTO MARKET_NEWS (
    TITLE, DESCRIPTION, URL, SOURCE_NAME, AUTHOR, PUBLISHED_AT,
    SENTIMENT, SENTIMENT_SCORE, KEY_ENTITY, KEY_ENTITY_TYPE, ONE_LINE_SUMMARY
)
SELECT %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
WHERE NOT EXISTS (
    SELECT 1 FROM MARKET_NEWS WHERE URL = %s
)
"""

# ---------------------------------------------------------------------------
# URL ALREADY EXISTS CHECK
# ---------------------------------------------------------------------------
URL_EXISTS_SQL = "SELECT COUNT(1) FROM MARKET_NEWS WHERE URL = %s"


def _normalize_account(raw_account: str) -> str:
    """
    Normalizes the Snowflake account identifier to the format expected
    by snowflake-connector-python: 'accountlocator.region'

    ETL CONCEPT — Defensive Configuration:
        The Snowflake web UI displays the account as 'us-east-1/asc63558'
        (region/locator). The Python connector requires the reverse:
        'asc63558.us-east-1' (locator.region).
        Normalizing here means the .env value works regardless of
        which format the user copied from their browser.

    Handles these input formats:
        'us-east-1/asc63558'   → 'asc63558.us-east-1'
        'asc63558.us-east-1'   → 'asc63558.us-east-1'  (already correct)
        'asc63558'             → 'asc63558'             (no region, use as-is)
    """
    # Strip inline comments (e.g., "asc63558  # example" → "asc63558")
    account = raw_account.split("#")[0].strip()

    if "/" in account:
        # Format from Snowflake web URL: 'region/locator' → flip to 'locator.region'
        parts = account.split("/")
        region, locator = parts[0].strip(), parts[1].strip()
        normalized = f"{locator}.{region}"
        logger.info(f"Account identifier normalized: '{account}' → '{normalized}'")
        return normalized

    # Already in correct format or just a locator
    return account


def _get_snowflake_connection() -> snowflake.connector.SnowflakeConnection:
    """
    Reads credentials from environment variables and opens a Snowflake connection.

    Returns:
        An active SnowflakeConnection object.

    Raises:
        ValueError: If any required credential is missing.
        snowflake.connector.errors.DatabaseError: If the connection fails.
    """
    required = [
        "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD", "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA",
    ]

    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise ValueError(
            f"Missing Snowflake credentials in .env: {', '.join(missing)}"
        )

    raw_account = os.getenv("SNOWFLAKE_ACCOUNT")
    account = _normalize_account(raw_account)

    logger.info(f"Connecting to Snowflake account: {account}...")

    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=account,
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )

    logger.info("Snowflake connection established successfully.")
    return conn


def _ensure_table_exists(cursor) -> None:
    """
    Creates the MARKET_NEWS table if it doesn't already exist.
    Safe to call on every pipeline run — does nothing if the table exists.
    """
    cursor.execute(CREATE_TABLE_SQL)
    logger.info("Table MARKET_NEWS is ready (created or already exists).")


def load_to_snowflake(records: list[dict]) -> int:
    """
    Main Load function. Inserts enriched records into Snowflake,
    skipping any that already exist (duplicate URL prevention).

    Args:
        records: List of enriched article dicts from transform_articles().

    Returns:
        The number of records successfully inserted.

    ETL CONCEPT — try/finally for connection cleanup:
        Database connections are expensive resources. We must close them
        whether the insert succeeds OR fails. The finally block guarantees
        cleanup no matter what happens — this prevents connection leaks
        that would exhaust your Snowflake connection pool over time.
    """
    if not records:
        logger.warning("No records to load. Skipping Snowflake insert.")
        return 0

    conn = None
    inserted_count = 0
    skipped_count = 0

    try:
        conn = _get_snowflake_connection()
        cursor = conn.cursor()

        # Ensure table exists before any inserts
        _ensure_table_exists(cursor)

        logger.info(f"Loading {len(records)} records into Snowflake...")

        for record in records:
            url = record.get("url", "")

            # --- Duplicate check ---
            # We check each URL before inserting. If it already exists,
            # we skip it rather than fail. This makes the pipeline IDEMPOTENT —
            # safe to re-run without creating duplicate rows.
            cursor.execute(URL_EXISTS_SQL, (url,))
            (count,) = cursor.fetchone()

            if count > 0:
                logger.info(f"  ⏭  Skipping duplicate: {record.get('title', '')[:60]}")
                skipped_count += 1
                continue

            # --- Insert the record ---
            # Note how the URL appears TWICE in the values tuple:
            # once for the INSERT columns, once for the WHERE NOT EXISTS check.
            cursor.execute(INSERT_SQL, (
                record.get("title", ""),
                record.get("description", ""),
                url,
                record.get("source_name", ""),
                record.get("author", ""),
                record.get("published_at", ""),
                record.get("sentiment", ""),
                record.get("sentiment_score", 0.5),
                record.get("key_entity", ""),
                record.get("key_entity_type", ""),
                record.get("one_line_summary", ""),
                url,   # for the WHERE NOT EXISTS subquery
            ))

            inserted_count += 1
            logger.info(f"  ✓  Inserted: {record.get('title', '')[:60]}")

        # Commit all inserts as a single transaction.
        # ETL CONCEPT — Atomicity: either ALL records in this batch commit,
        # or NONE do. If something crashes mid-batch, the transaction rolls
        # back automatically, leaving the table in a consistent state.
        conn.commit()

        logger.info(
            f"Load complete. "
            f"{inserted_count} inserted / {skipped_count} skipped (duplicates)."
        )

    except snowflake.connector.errors.ProgrammingError as e:
        logger.error(f"Snowflake SQL error: {e}")
        if conn:
            conn.rollback()
        raise

    except snowflake.connector.errors.DatabaseError as e:
        logger.error(f"Snowflake connection/database error: {e}")
        raise

    finally:
        # Always close the connection — success or failure
        if conn:
            conn.close()
            logger.info("Snowflake connection closed.")

    return inserted_count


# ---------------------------------------------------------------------------
# Quick test — run this file directly to verify the full ETL works
# Usage: python -m src.load
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.extract import fetch_market_news
    from src.transform import transform_articles

    logger.info("=== Running Full ETL Pipeline Test ===")

    logger.info("Step 1: Extract...")
    raw = fetch_market_news()

    logger.info("Step 2: Transform...")
    enriched = transform_articles(raw)

    logger.info("Step 3: Load...")
    count = load_to_snowflake(enriched)

    print("\n" + "=" * 55)
    print("  PIPELINE RUN SUMMARY")
    print("=" * 55)
    print(f"  Articles extracted  : {len(raw)}")
    print(f"  Articles enriched   : {len(enriched)}")
    print(f"  Records inserted    : {count}")
    print(f"  Records skipped     : {len(enriched) - count} (duplicates)")
    print("=" * 55)

    if count > 0:
        print("\n  ✅  Load layer working. Check your Snowflake table:")
        print("      SELECT * FROM MARKET_INTEL.RAW.MARKET_NEWS ORDER BY INGESTED_AT DESC;\n")
    else:
        print("\n  ⚠️   0 records inserted — all may already exist from a previous run.\n")
