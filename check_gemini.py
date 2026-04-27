"""
check_gemini.py
---------------
Diagnostic script — run this to see exactly which Gemini models
are available for your current API key, and test a live call.

Usage:
    python check_gemini.py
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY", "").strip("'\"")

if not api_key or api_key.startswith("your_"):
    print("\n❌  GEMINI_API_KEY is not set in your .env file.\n")
    exit(1)

print(f"\n🔑  Using API key: {api_key[:8]}...{api_key[-4:]}")
print(f"    (If this doesn't look like your personal Gmail key, update .env)\n")

genai.configure(api_key=api_key)

# --- Step 1: List all available models ---
print("=" * 55)
print("  MODELS AVAILABLE FOR THIS API KEY")
print("=" * 55)

available = []
try:
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            available.append(m.name)
            print(f"  ✓  {m.name}")
except Exception as e:
    print(f"  ❌  Could not list models: {e}")
    exit(1)

if not available:
    print("\n  No models found. Your API key may be invalid or restricted.")
    exit(1)

# --- Step 2: Try a quick test call on the best available model ---
# Prefer flash-lite → flash → pro, in that order
preferred = [
    "models/gemini-2.0-flash-lite",
    "models/gemini-2.0-flash",
    "models/gemini-1.5-flash-latest",
    "models/gemini-1.5-flash",
    "models/gemini-1.5-pro",
    "models/gemini-pro",
]

test_model_name = next((m for m in preferred if m in available), available[0])
print(f"\n  → Will test with: {test_model_name}\n")

print("=" * 55)
print("  LIVE API TEST")
print("=" * 55)

try:
    model = genai.GenerativeModel(test_model_name)
    response = model.generate_content(
        'Return only this JSON: {"status": "ok", "message": "Gemini is working"}'
    )
    print(f"\n  ✅  Success! Response: {response.text.strip()}")
    print(f"\n  👉  Use this model name in transform.py:")
    print(f"      GEMINI_MODEL = \"{test_model_name.replace('models/', '')}\"")
except Exception as e:
    print(f"\n  ❌  Test call failed: {e}")

print()
