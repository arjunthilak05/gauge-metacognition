#!/usr/bin/env python3
"""
Quick model validation script.
Tests each model individually to identify JSON parsing errors or other failures.
Useful before running the full benchmark to catch problematic models early.

Usage:
  python test_models.py

Output:
  - Model test results (pass/fail/error)
  - JSON error details (if any)
  - Recommended model list for metacog_full.py
"""

import sys
import json

try:
    import kaggle_benchmarks as kbench
except ImportError:
    print("ERROR: kaggle_benchmarks SDK not installed.")
    print("Install with: pip install kaggle-benchmarks")
    sys.exit(1)

# Models to test (in priority order)
MODELS_TO_TEST = [
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "google/gemini-3-flash-preview",
    "google/gemma-3-27b",
    "anthropic/claude-sonnet-4@20250514",
    "anthropic/claude-haiku-4-5@20251001",
    "qwen/qwen3-235b-a22b-instruct-2507",
    "deepseek-ai/deepseek-v3.1",
]

FALLBACK_MODELS = [
    "zai/glm-5",
    "qwen/qwen3-next-80b-a3b-instruct",
    "google/gemini-3-pro-preview",
]

def test_model_availability(model_id: str) -> tuple[bool, str]:
    """
    Test if a model is available in kbench.llms

    Returns: (available, error_msg)
    """
    try:
        llm = kbench.llms[model_id]
        return True, f"✅ Available ({type(llm).__name__})"
    except KeyError:
        return False, f"❌ Not found in kbench.llms"
    except Exception as e:
        return False, f"❌ Error accessing: {type(e).__name__}: {str(e)[:60]}"


def test_model_simple_prompt(model_id: str) -> tuple[bool, str]:
    """
    Test if a model can handle a simple prompt without crashing.
    This catches JSON parsing errors and other runtime issues.

    Returns: (success, result_msg)
    """
    try:
        llm = kbench.llms[model_id]
        # Simple non-structured prompt (avoids Pydantic schema issues)
        response = llm.prompt("What is 2+2? Answer with just the number.")
        return True, f"✅ Simple prompt OK (response: {str(response)[:40]}...)"
    except json.JSONDecodeError as e:
        return False, f"❌ JSON Parsing Error: {str(e)[:80]}"
    except TypeError as e:
        if "JSON" in str(e) or "json" in str(e):
            return False, f"❌ JSON Type Error: {str(e)[:80]}"
        return False, f"❌ Type Error: {str(e)[:80]}"
    except Exception as e:
        error_str = str(e)
        # Check for specific error patterns
        if "<think>" in error_str or "</think>" in error_str:
            return False, f"❌ CoT Output Error (model outputs <think> tags): {str(e)[:60]}"
        return False, f"❌ {type(e).__name__}: {str(e)[:80]}"


def main():
    print("=" * 70)
    print("  MODEL VALIDATION TEST")
    print("=" * 70)
    print(f"\nTesting {len(MODELS_TO_TEST)} primary models...\n")

    results = {
        'available': [],
        'unavailable': [],
        'error': [],
    }

    for idx, model_id in enumerate(MODELS_TO_TEST, 1):
        print(f"[{idx}/{len(MODELS_TO_TEST)}] {model_id:<50}", end=" ")

        # Test availability
        available, avail_msg = test_model_availability(model_id)
        if not available:
            print(avail_msg)
            results['unavailable'].append((model_id, avail_msg))
            continue

        # Test simple prompt
        success, prompt_msg = test_model_simple_prompt(model_id)
        print(prompt_msg)

        if success:
            results['available'].append(model_id)
        else:
            results['error'].append((model_id, prompt_msg))

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    print(f"\n✅ AVAILABLE & WORKING ({len(results['available'])}):")
    for model in results['available']:
        print(f"   • {model}")

    if results['unavailable']:
        print(f"\n⚠️  NOT FOUND ({len(results['unavailable'])}):")
        for model, msg in results['unavailable']:
            print(f"   • {model}: {msg}")

    if results['error']:
        print(f"\n❌ ERROR ({len(results['error'])}):")
        for model, msg in results['error']:
            print(f"   • {model}: {msg}")

    # Recommendation
    print(f"\n" + "=" * 70)
    print("  RECOMMENDATION FOR metacog_full.py")
    print("=" * 70)

    if results['available']:
        print(f"\nWorking models ({len(results['available'])}): UPDATE ALL_MODELS to:")
        print("\nALL_MODELS = [")
        for model in results['available']:
            print(f"    kbench.llms[\"{model}\"],")
        print("]")

    if results['error']:
        print(f"\n⚠️  REMOVE these models (JSON/output parsing errors):")
        for model, msg in results['error']:
            print(f"   - {model}")
            print(f"     {msg}\n")

        print(f"💡 FALLBACK models to try instead:")
        for fallback in FALLBACK_MODELS:
            if fallback not in results['unavailable']:
                print(f"   - {fallback}")

    return len(results['error']) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
