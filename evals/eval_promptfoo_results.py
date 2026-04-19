"""
Promptfoo Evaluation Runner for SL-RAG / Trust-RAG Pipeline.

Runs the Promptfoo red-teaming eval defined in promptfooconfig.yaml,
then normalises the raw CLI output into eval_promptfoo_results.json
for use by eval_visualize.py.

Usage:
    python eval_promptfoo_results.py

Requirements:
    Node.js 20+ with promptfoo installed globally:
        npm install -g promptfoo

    LM Studio must be running at http://localhost:1234/v1

    promptfooconfig.yaml must exist in the current directory.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ── Checks ─────────────────────────────────────────────────────────────────────
if not Path("promptfooconfig.yaml").exists():
    print("[ERROR] promptfooconfig.yaml not found. Run from project root.")
    sys.exit(1)

promptfoo_bin = shutil.which("promptfoo")
if not promptfoo_bin:
    print("[ERROR] promptfoo not found in PATH.")
    print("        Install with: npm install -g promptfoo")
    sys.exit(1)

print(f"[PROMPTFOO] Using promptfoo at: {promptfoo_bin}")

# LM Studio check
LM_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
try:
    from openai import OpenAI as _OAI
    _probe = _OAI(base_url=LM_BASE_URL, api_key="lm-studio")
    _probe.models.list()
    print(f"[PROMPTFOO] LM Studio reachable at {LM_BASE_URL} ✓")
except Exception as e:
    print(f"[WARN] Cannot reach LM Studio at {LM_BASE_URL}: {e}")
    print("       Promptfoo will still run; tests may fail due to provider error.")

# ── Run promptfoo eval ─────────────────────────────────────────────────────────
RAW_OUTPUT = "eval_promptfoo_raw.json"

print("\n[PROMPTFOO] Running: promptfoo eval --config promptfooconfig.yaml ...")
print("            This may take 5–15 minutes.\n")

t_start = time.time()
eval_env = os.environ.copy()
# Route rubric grading to local LM Studio instead of cloud defaults.
eval_env.setdefault("OPENAI_API_KEY", "lm-studio")
eval_env.setdefault("OPENAI_BASE_URL", LM_BASE_URL)
result = subprocess.run(
    [
        promptfoo_bin, "eval",
        "--config", "promptfooconfig.yaml",
        "--output", RAW_OUTPUT,
        "--no-cache",        # always fresh results
        "--max-concurrency", "1",
        "--grader", "openai:chat:google/gemma-4-e4b",
    ],
    env=eval_env,
    capture_output=False,   # let output stream to terminal
    text=True,
)
elapsed = round(time.time() - t_start, 1)

if result.returncode != 0:
    print(f"\n[WARN] promptfoo exited with code {result.returncode}")
    print("       Attempting to parse partial output if available...")

# ── Parse and normalise raw output ────────────────────────────────────────────
if not Path(RAW_OUTPUT).exists():
    print(f"[ERROR] {RAW_OUTPUT} was not created. Promptfoo may have failed.")
    print("        Check the output above for errors.")
    sys.exit(1)

with open(RAW_OUTPUT, encoding="utf-8") as f:
    raw = json.load(f)

print(f"\n[PROMPTFOO] Parsing raw output from {RAW_OUTPUT}...")

# Promptfoo JSON structure varies by version; handle common shapes
def _extract_results(raw_data: dict) -> list:
    """Return a flat list of test result dicts from promptfoo JSON output."""
    # v0.x shape: {"results": [...]}
    if "results" in raw_data and isinstance(raw_data["results"], list):
        return raw_data["results"]
    # v1.x shape: {"results": {"results": [...]}}
    if "results" in raw_data and isinstance(raw_data["results"], dict):
        inner = raw_data["results"]
        if "results" in inner:
            return inner["results"]
        if "table" in inner:
            return inner.get("table", {}).get("body", [])
    return []


raw_results = _extract_results(raw)

normalized_tests = []
pass_count       = 0
fail_count       = 0
adversarial_pass = 0
adversarial_total = 0

for entry in raw_results:
    # Support both flat and nested result shapes
    desc        = (
        entry.get("description")
        or entry.get("testCase", {}).get("description")
        or entry.get("prompt", {}).get("description")
        or ""
    )
    test_input  = (entry.get("vars") or {}).get("question") or entry.get("prompt", {}).get("raw") or ""
    response    = entry.get("response", {})
    output_text = response.get("output") or response.get("text") or ""
    error_text  = entry.get("error") or response.get("error") or ""

    # Assertion details
    assert_results = entry.get("assertResults") or entry.get("namedScores") or []
    assertions = []
    if isinstance(assert_results, list):
        for ar in assert_results:
            assertions.append({
                "label":  ar.get("assertion", {}).get("label") or ar.get("label") or "",
                "passed": ar.get("pass") if "pass" in ar else ar.get("passed"),
                "score":  ar.get("score"),
                "reason": ar.get("reason") or ar.get("failureReason") or "",
            })
    elif isinstance(assert_results, dict):
        for label, score in assert_results.items():
            assertions.append({"label": label, "score": score})

    if "success" in entry:
        passed = bool(entry.get("success"))
    elif "passed" in entry:
        passed = bool(entry.get("passed"))
    else:
        passed = all(a.get("passed", True) for a in assertions) if assertions else True

    if error_text:
        passed = False

    is_adversarial = "ADVERSARIAL" in desc.upper()
    if is_adversarial:
        adversarial_total += 1
        if passed:
            adversarial_pass += 1

    if passed:
        pass_count += 1
    else:
        fail_count += 1

    normalized_tests.append({
        "description":   desc,
        "question":      test_input,
        "is_adversarial": is_adversarial,
        "passed":        passed,
        "output":        output_text[:600] if output_text else "",
        "error":         error_text[:600] if error_text else "",
        "assertions":    assertions,
    })

total = pass_count + fail_count
pass_rate           = round(pass_count / max(total, 1), 4)
adversarial_pass_rate = round(adversarial_pass / max(adversarial_total, 1), 4)

output = {
    "eval_timestamp":          time.strftime("%Y-%m-%dT%H:%M:%S"),
    "eval_time_seconds":       elapsed,
    "promptfoo_exit_code":     result.returncode,
    "total_tests":             total,
    "pass_count":              pass_count,
    "fail_count":              fail_count,
    "pass_rate":               pass_rate,
    "adversarial_total":       adversarial_total,
    "adversarial_pass_count":  adversarial_pass,
    "adversarial_pass_rate":   adversarial_pass_rate,
    "tests":                   normalized_tests,
}

with open("eval_promptfoo_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 58)
print("  PROMPTFOO SUMMARY")
print("=" * 58)
print(f"  Total tests         : {total}")
print(f"  Passed              : {pass_count} ({pass_rate:.1%})")
print(f"  Failed              : {fail_count}")
print(f"  Adversarial tests   : {adversarial_total}")
print(f"  Adversarial passed  : {adversarial_pass} ({adversarial_pass_rate:.1%})")
print(f"\n  Eval time           : {elapsed}s")
print("=" * 58)
print("\n[PROMPTFOO] Normalised results saved → eval_promptfoo_results.json")
