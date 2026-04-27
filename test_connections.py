"""
test_connections.py
-------------------
PURPOSE: Verify that all required environment variables are loaded correctly
         from the .env file before any live API or database connections are made.

ETL CONCEPT: In enterprise pipelines, you always validate your configuration
             at startup ("fail fast"). If credentials are missing, you want to
             know immediately — not halfway through a pipeline run.

USAGE:
    python test_connections.py
"""

import os
import sys
from dotenv import load_dotenv

# --- Load .env file ---
# load_dotenv() looks for a .env file in the current directory and injects
# all key=value pairs into the process's environment variables (os.environ).
# This keeps secrets OUT of source code.
load_dotenv()


def check_env_variables() -> bool:
    """
    Validates that all required environment variables are present and non-empty.
    Returns True if all checks pass, False otherwise.
    """

    required_vars = {
        "News API":   ["NEWS_API_KEY"],
        "Gemini API": ["GEMINI_API_KEY"],
        "Snowflake":  [
            "SNOWFLAKE_USER",
            "SNOWFLAKE_PASSWORD",
            "SNOWFLAKE_ACCOUNT",
            "SNOWFLAKE_WAREHOUSE",
            "SNOWFLAKE_DATABASE",
            "SNOWFLAKE_SCHEMA",
        ],
    }

    all_passed = True
    print("\n" + "=" * 55)
    print("  Automated Market Intelligence — Environment Check")
    print("=" * 55)

    for service, variables in required_vars.items():
        print(f"\n[{service}]")
        for var in variables:
            value = os.getenv(var)
            if value and value.strip() and not value.startswith("your_"):
                # Mask the value for security — never print raw secrets
                masked = value[:4] + "*" * (len(value) - 4)
                print(f"  ✓  {var:<30} → {masked}")
            else:
                print(f"  ✗  {var:<30} → NOT SET or still a placeholder")
                all_passed = False

    print("\n" + "=" * 55)
    if all_passed:
        print("  ✅  All environment variables loaded successfully.")
        print("  Your local environment is secure and ready.")
    else:
        print("  ❌  One or more variables are missing or still placeholders.")
        print("  Open your .env file and fill in the real values.")
    print("=" * 55 + "\n")

    return all_passed


if __name__ == "__main__":
    success = check_env_variables()
    # Exit with a non-zero code on failure — important for CI/CD pipelines
    # so GitHub Actions can detect a broken environment automatically.
    sys.exit(0 if success else 1)
