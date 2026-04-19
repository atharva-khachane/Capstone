"""
DeepEval Evaluation for SL-RAG + Trust-RAG Pipeline.

Reads pipeline outputs from eval_paper_results.json (produced by eval_extended.py)
and runs DeepEval metrics using a configurable judge LLM provider.

Metrics computed:
    1. HallucinationMetric  — claims not grounded in retrieved context
    2. G-Eval Faithfulness  — chain-of-thought faithfulness scoring
    3. G-Eval Correctness   — factual correctness vs ground truth
    4. AnswerRelevancyMetric — how directly the answer addresses the question
    5. ContextualPrecision  — fraction of retrieved chunks that were useful
    6. ContextualRecall     — ground-truth information coverage in retrieved context

Usage:
    python eval_deepeval.py

Requirements:
    pip install deepeval groq openai
    Configure one judge provider:
      - Groq: set GROQ_API_KEY (or JUDGE_API_KEY)
      - OpenAI-compatible: set DEEPINFRA_API_KEY / OPENAI_API_KEY (or JUDGE_API_KEY)
            - W&B Inference: set WANDB_API_KEY (or JUDGE_API_KEY),
                JUDGE_BASE_URL=https://api.inference.wandb.ai/v1
"""
# ── stdlib imports FIRST — env vars must come after `import os` ───────────────
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Environment configuration (must be set BEFORE deepeval is imported) ────────
# Bug 2 fix: ENTAILMENT_SYNC=true forces DeepEval to use the synchronous NLI
# pipeline instead of the async variant that systematically mis-scores entailment
# and inflated hallucination to 0.338 in Run 6.
os.environ["ENTAILMENT_SYNC"] = "true"
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

assert os.environ.get("ENTAILMENT_SYNC") == "true", (
    "ENTAILMENT_SYNC must be 'true' — set it before DeepEval is imported "
    "or hallucination scores will be wrong (Run 6 regression: 0.338)."
)

print(
    f"[DEEPEVAL] Env config: "
    f"ENTAILMENT_SYNC={os.environ['ENTAILMENT_SYNC']}  "
    f"DEEPEVAL_TELEMETRY_OPT_OUT={os.environ['DEEPEVAL_TELEMETRY_OPT_OUT']}"
)

# ── Validate prerequisite ─────────────────────────────────────────────────────
if not Path("eval_paper_results.json").exists():
    print("[ERROR] eval_paper_results.json not found.")
    print("        Run eval_extended.py first to generate pipeline outputs.")
    sys.exit(1)

with open("eval_paper_results.json", encoding="utf-8") as f:
    paper_results = json.load(f)

per_query = paper_results.get("per_query", [])
if not per_query:
    print("[ERROR] No per_query records found in eval_paper_results.json")
    sys.exit(1)

print(f"[DEEPEVAL] Loaded {len(per_query)} queries from eval_paper_results.json")

# Optional focused run on specific IDs, e.g. EVAL_ONLY_IDS=Q004,Q005,Q006
# _only_ids_raw = os.getenv("EVAL_ONLY_IDS", "").strip()
# if _only_ids_raw:
#     only_ids = [x.strip() for x in _only_ids_raw.split(",") if x.strip()]
#     if only_ids:
#         wanted = set(only_ids)
#         per_query = [item for item in per_query if item.get("id") in wanted]
#         found = {item.get("id") for item in per_query}
#         missing = [qid for qid in only_ids if qid not in found]
#         print(f"[DEEPEVAL] Focus mode: selected {len(per_query)} IDs via EVAL_ONLY_IDS")
#         if missing:
#             print(f"[WARN] Missing IDs in dataset: {', '.join(missing)}")

# Optional one-time pilot limiter (0 means evaluate all queries)
EVAL_MAX_SAMPLES = int(os.getenv("EVAL_MAX_SAMPLES", "0"))
if EVAL_MAX_SAMPLES > 0:
    per_query = per_query[:EVAL_MAX_SAMPLES]
    print(f"[DEEPEVAL] Pilot mode: limiting to first {len(per_query)} samples")

# ── Judge provider configuration ───────────────────────────────────────────────
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "groq").strip().lower()
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "").strip()
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY", "").strip()
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "").strip()

if JUDGE_PROVIDER == "groq":
    ACTIVE_MODEL = JUDGE_MODEL or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    ACTIVE_API_KEY = JUDGE_API_KEY or os.getenv("GROQ_API_KEY", "")
    ACTIVE_BASE_URL = None
    if not ACTIVE_API_KEY:
        print("[ERROR] GROQ_API_KEY (or JUDGE_API_KEY) is not set.")
        sys.exit(1)
elif JUDGE_PROVIDER in {"openai", "openai_compatible", "deepinfra", "wandb"}:
    ACTIVE_MODEL = (
        JUDGE_MODEL
        or os.getenv("WANDB_MODEL", "")
        or os.getenv("DEEPINFRA_MODEL", "")
        or os.getenv("OPENAI_MODEL", "")
        or "meta-llama/Llama-3.3-70B-Instruct"
    )
    ACTIVE_API_KEY = (
        JUDGE_API_KEY
        or os.getenv("WANDB_API_KEY", "")
        or os.getenv("DEEPINFRA_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
    )
    if JUDGE_PROVIDER == "wandb":
        ACTIVE_BASE_URL = (
            JUDGE_BASE_URL
            or os.getenv("WANDB_BASE_URL", "")
            or os.getenv("OPENAI_BASE_URL", "")
            or "https://api.inference.wandb.ai/v1"
        )
    else:
        ACTIVE_BASE_URL = (
            JUDGE_BASE_URL
            or os.getenv("DEEPINFRA_BASE_URL", "")
            or os.getenv("OPENAI_BASE_URL", "")
            or "https://api.deepinfra.com/v1/openai"
        )
    if not ACTIVE_API_KEY:
        print(
            "[ERROR] Set one of: JUDGE_API_KEY / WANDB_API_KEY / "
            "DEEPINFRA_API_KEY / OPENAI_API_KEY."
        )
        sys.exit(1)
else:
    print(
        "[ERROR] Unsupported JUDGE_PROVIDER. Use one of: "
        "groq, openai, openai_compatible, deepinfra, wandb"
    )
    sys.exit(1)

if JUDGE_PROVIDER == "groq":
    print(f"[DEEPEVAL] Judge provider: groq  model: {ACTIVE_MODEL} [OK]")
else:
    print(
        f"[DEEPEVAL] Judge provider: {JUDGE_PROVIDER}  model: {ACTIVE_MODEL}  "
        f"base_url: {ACTIVE_BASE_URL} [OK]"
    )

# Groq needs conservative pacing by default; OpenAI-compatible providers generally do not.
default_delay = "12" if JUDGE_PROVIDER == "groq" else "0"
PER_METRIC_DELAY_SECONDS = float(os.getenv("METRIC_CALL_DELAY_S", default_delay))
USE_JSON_RESPONSE_FORMAT = os.getenv("JUDGE_JSON_MODE", "true").strip().lower() == "true"

# ── DeepEval imports ──────────────────────────────────────────────────────────
try:
    from deepeval import evaluate
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from deepeval.metrics import (
        HallucinationMetric,
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        GEval,
    )
    from deepeval.models.base_model import DeepEvalBaseLLM
except ImportError as e:
    print(f"[ERROR] DeepEval not installed: {e}")
    print("        Run: pip install deepeval")
    sys.exit(1)

# ── Judge model wrapper ───────────────────────────────────────────────────────
class JudgeModel(DeepEvalBaseLLM):
    """Wraps either Groq or OpenAI-compatible providers for DeepEval."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str,
        base_url: str | None = None,
    ):
        self._provider = provider
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._timeout_s = float(os.getenv("JUDGE_TIMEOUT_S", "90"))
        self._max_tokens = int(os.getenv("JUDGE_MAX_TOKENS", "1024"))
        self._debug_schema = os.getenv("DEEPEVAL_DEBUG_SCHEMA", "0").strip() == "1"
        self._rpm_limit = int(os.getenv("JUDGE_RPM_LIMIT", "0"))
        self._rpm_window_s = 60.0
        self._recent_calls: list[float] = []

    def _throttle_if_needed(self):
        """Client-side RPM throttling to avoid provider-side 429s."""
        if self._rpm_limit <= 0:
            return

        now = time.time()
        cutoff = now - self._rpm_window_s
        self._recent_calls = [t for t in self._recent_calls if t >= cutoff]

        if len(self._recent_calls) >= self._rpm_limit:
            oldest = self._recent_calls[0]
            wait = max(0.0, self._rpm_window_s - (now - oldest) + 0.05)
            if wait > 0:
                print(f"\n    [THROTTLE] Client RPM limit reached ({self._rpm_limit}/min). Waiting {wait:.1f}s...")
                time.sleep(wait)

            now = time.time()
            cutoff = now - self._rpm_window_s
            self._recent_calls = [t for t in self._recent_calls if t >= cutoff]

        self._recent_calls.append(time.time())

    def load_model(self):
        if self._provider == "groq":
            from groq import Groq

            return Groq(api_key=self._api_key)

        from openai import OpenAI

        return OpenAI(base_url=self._base_url, api_key=self._api_key, timeout=self._timeout_s)

    @staticmethod
    def _schema_to_text(schema_obj) -> str:
        """Best-effort schema serialization for instruction prompts."""
        if schema_obj is None:
            return ""
        try:
            if hasattr(schema_obj, "model_json_schema"):
                return json.dumps(schema_obj.model_json_schema(), ensure_ascii=False)
            if hasattr(schema_obj, "schema"):
                return json.dumps(schema_obj.schema(), ensure_ascii=False)
        except Exception:
            pass
        return str(schema_obj)

    @staticmethod
    def _extract_first_json_object(text: str) -> str:
        """Extract the first valid JSON object from model output if possible."""
        if not text:
            return text

        import re

        cleaned = text.strip()
        # Strip markdown code fences: ```json ... ``` or ``` ... ```
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        # Fast path: already strict JSON
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            pass

        # Fallback: bracket-balanced extraction
        start = cleaned.find("{")
        if start == -1:
            return cleaned
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return json.dumps(parsed, ensure_ascii=False)
                    except Exception:
                        break
        return cleaned

    @staticmethod
    def _schema_accepts(schema_obj, json_text: str) -> bool:
        """Return True if json_text validates against the provided schema object."""
        if schema_obj is None:
            return True
        try:
            if hasattr(schema_obj, "model_validate_json"):
                schema_obj.model_validate_json(json_text)
                return True
            if hasattr(schema_obj, "parse_raw"):
                schema_obj.parse_raw(json_text)
                return True
        except Exception:
            return False
        return True

    @classmethod
    def _coerce_to_schema_json(cls, content: str, schema_obj) -> str:
        """Coerce arbitrary model output into JSON that validates against schema_obj.

        Returns an empty string when coercion fails; caller should treat as metric failure.
        """
        raw = (content or "").strip()
        if not raw:
            return ""

        # Remove common markdown fencing wrappers.
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json\n", "", 1).strip()

        candidates = []
        candidates.append(raw)
        first_obj = cls._extract_first_json_object(raw)
        if first_obj != raw:
            candidates.append(first_obj)

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    normalized = json.dumps(parsed, ensure_ascii=False)
                    if cls._schema_accepts(schema_obj, normalized):
                        return normalized
            except Exception:
                continue

        return ""

    def generate(self, prompt: str, *args, **kwargs) -> str:
        client = self.load_model()
        schema = kwargs.get("schema")

        # When DeepEval requests schema output, reinforce strict JSON behavior.
        if schema is not None:
            schema_text = self._schema_to_text(schema)
            prompt = (
                f"{prompt}\n\n"
                "Return ONLY a valid JSON object. "
                "Do not include markdown/code fences or extra commentary.\n"
                f"JSON schema reference: {schema_text}"
            )

        max_retries = 8
        for attempt in range(max_retries):
            try:
                request_kwargs = {
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": self._max_tokens,
                }

                if schema is not None and self._provider != "groq" and USE_JSON_RESPONSE_FORMAT:
                    # Supported by many OpenAI-compatible endpoints.
                    request_kwargs["response_format"] = {"type": "json_object"}

                self._throttle_if_needed()
                resp = client.chat.completions.create(**request_kwargs)
                content = resp.choices[0].message.content or ""
                finish_reason = getattr(resp.choices[0], "finish_reason", None)

                # If structured output was truncated, retry once with a larger token budget.
                if schema is not None and finish_reason == "length":
                    retry_kwargs = dict(request_kwargs)
                    retry_kwargs["max_tokens"] = max(self._max_tokens * 2, 1200)
                    self._throttle_if_needed()
                    resp2 = client.chat.completions.create(**retry_kwargs)
                    content = resp2.choices[0].message.content or ""
                if schema is not None:
                    coerced = self._coerce_to_schema_json(content, schema)
                    if not coerced:
                        if self._debug_schema:
                            snippet = content[:350].replace("\n", " ")
                            print(f"\n    [DEBUG] Non-schema output snippet: {snippet}")
                        return (
                            "[schema_parse_error: judge returned non-JSON or schema-incompatible JSON]"
                        )
                    return coerced
                return content
            except Exception as e:
                import re

                err = str(e).lower()
                is_rate_limit = "429" in str(e) or "rate_limit" in err or "rate limit" in err
                is_timeout = "timed out" in err or "timeout" in err or "read operation" in err
                if not is_rate_limit and not is_timeout:
                    return f"[{self._provider} error: {e}]"
                is_tpm = "token" in err or "tpm" in err
                m = re.search(r"retry.after[^\d]*(\d+)", err)
                retry_after = int(m.group(1)) + 2 if m else None
                if retry_after:
                    wait = retry_after
                elif is_timeout:
                    wait = min(8 * (attempt + 1), 45)
                elif is_tpm:
                    wait = 60
                else:
                    wait = min(15 * (2 ** attempt), 120)
                limit_type = "TIMEOUT" if is_timeout else ("TPM" if is_tpm else "RPM")
                print(
                    f"\n    [RATE LIMIT] {limit_type} hit — waiting {wait}s "
                    f"(attempt {attempt+1}/{max_retries})..."
                )
                time.sleep(wait)
        return f"[{self._provider} error: max retries exceeded]"

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        return self.generate(prompt, *args, **kwargs)

    def get_model_name(self) -> str:
        if self._provider == "groq":
            return f"Groq ({self._model})"
        return f"{self._provider} ({self._model})"


judge = JudgeModel(
    provider=JUDGE_PROVIDER,
    api_key=ACTIVE_API_KEY,
    model=ACTIVE_MODEL,
    base_url=ACTIVE_BASE_URL,
)
print(f"[DEEPEVAL] Judge model  : {judge.get_model_name()}")
print(f"[DEEPEVAL] Metrics     : hallucination, answer_relevancy, contextual_precision, "
      f"contextual_recall, faithfulness_geval, answer_correctness_geval")

# ── Metrics definition ────────────────────────────────────────────────────────
hallucination_metric = HallucinationMetric(
    threshold=0.5,
    model=judge,
    include_reason=True,
)

answer_relevancy_metric = AnswerRelevancyMetric(
    threshold=0.5,
    model=judge,
    include_reason=True,
)

contextual_precision_metric = ContextualPrecisionMetric(
    threshold=0.5,
    model=judge,
    include_reason=True,
)

contextual_recall_metric = ContextualRecallMetric(
    threshold=0.5,
    model=judge,
    include_reason=True,
)

faithfulness_geval = GEval(
    name="Faithfulness",
    criteria=(
        "Determine whether the ACTUAL OUTPUT is fully supported by the information "
        "in the RETRIEVAL CONTEXT. An output is faithful if every factual claim it "
        "makes can be directly traced back to the provided context. Score 1 if fully "
        "faithful, 0 if it introduces unsupported claims."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
    threshold=0.5,
    model=judge,
)

correctness_geval = GEval(
    name="Answer Correctness",
    criteria=(
        "Assess whether the ACTUAL OUTPUT correctly answers the INPUT question "
        "compared to the EXPECTED OUTPUT (ground truth). Consider factual accuracy, "
        "completeness of key points, and absence of contradictions. "
        "Score 1 for fully correct, 0 for incorrect or missing critical information."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.5,
    model=judge,
)

# Each entry: (metric_object, result_key, aggregate_key)
all_metrics = [
    (hallucination_metric,         "hallucination",              "hallucination"),
    (answer_relevancy_metric,      "answer_relevancy",           "answer_relevancy"),
    (contextual_precision_metric,  "contextual_precision",       "contextual_precision"),
    (contextual_recall_metric,     "contextual_recall",          "contextual_recall"),
    (faithfulness_geval,           "faithfulness_geval",         "faithfulness_geval"),
    (correctness_geval,            "answer_correctness_geval",   "answer_correctness_geval"),
]

# ── Build test cases ──────────────────────────────────────────────────────────
print(f"\n[DEEPEVAL] Building {len(per_query)} test cases...")
test_cases = []
for item in per_query:
    answer   = item.get("answer") or ""
    contexts = item.get("contexts") or []

    # Skip entries where the LLM failed to generate an answer
    if not answer or answer.startswith("["):
        contexts = contexts or ["No context available."]
        answer   = answer or "No answer generated."

    tc = LLMTestCase(
        input=item["question"],
        actual_output=answer,
        expected_output=item["ground_truth"],
        retrieval_context=contexts,
        context=contexts,
    )
    test_cases.append(tc)

# ── Evaluate ──────────────────────────────────────────────────────────────────
print(f"[DEEPEVAL] Evaluating {len(test_cases)} test cases with 6 metrics...")
print("           Duration depends on provider latency and rate limits.\n")

per_sample_results = []
aggregates: dict = {
    "hallucination":          [],
    "answer_relevancy":       [],
    "contextual_precision":   [],
    "contextual_recall":      [],
    "faithfulness_geval":     [],
    "answer_correctness_geval": [],
}

t_start = time.time()
try:
    for i, (tc, item) in enumerate(zip(test_cases, per_query), 1):
        qid = item.get("id", f"Q{i:03d}")
        print(f"  [{i}/{len(test_cases)}] QID={qid}...", end="", flush=True)
        sample_scores: dict = {"id": qid, "question": item["question"]}

        for metric, key, agg_key in all_metrics:
            try:
                metric.measure(tc)
                score = metric.score
                sample_scores[key] = round(float(score), 4) if score is not None else None
                if score is not None and agg_key in aggregates:
                    aggregates[agg_key].append(float(score))
            except Exception as e:
                sample_scores[key] = None
                print(f"\n    [WARN] QID={qid} {key} failed: {str(e)[:80]}")
            finally:
                if PER_METRIC_DELAY_SECONDS > 0:
                    time.sleep(PER_METRIC_DELAY_SECONDS)

        per_sample_results.append(sample_scores)
        agg_str = " ".join(
            f"{k[:4]}={v:.2f}" for k, v in sample_scores.items()
            if isinstance(v, float)
        )
        print(f" {agg_str}")
except KeyboardInterrupt:
    print("\n[WARN] Evaluation interrupted by user. Saving partial results...")

eval_time = round(time.time() - t_start, 1)

# Aggregate
aggregate_summary = {}
for k, vals in aggregates.items():
    aggregate_summary[k] = round(sum(vals) / len(vals), 4) if vals else None

# ── Save ──────────────────────────────────────────────────────────────────────
output = {
    "eval_timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
    "judge_model":     judge.get_model_name(),
    "target_samples":  len(test_cases),
    "processed_samples": len(per_sample_results),
    "num_samples":     len(per_sample_results),
    "eval_time_seconds": eval_time,
    "aggregate":       aggregate_summary,
    "per_sample":      per_sample_results,
}

with open("eval_deepeval_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  DEEPEVAL SUMMARY  (Judge: {judge.get_model_name()})")
print("=" * 60)
for k, v in aggregate_summary.items():
    label = k.replace("_", " ").title()
    print(f"  {label:<40}: {v}")
print(f"\n  Samples processed : {len(per_sample_results)} / {len(test_cases)}")
print(f"  Eval time         : {eval_time}s")
print("=" * 60)
print("\n[DEEPEVAL] Results saved → eval_deepeval_results.json")
