import importlib.util
from pathlib import Path


_POLICY_PATH = Path(__file__).resolve().parent.parent / "sl_rag" / "retrieval" / "policy.py"
_SPEC = importlib.util.spec_from_file_location("sl_rag_retrieval_policy", _POLICY_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_MODULE)
RetrievalPolicy = _MODULE.RetrievalPolicy


def test_retrieval_policy_defaults_match_spec():
    policy = RetrievalPolicy()
    assert policy.similarity_threshold == 0.5
    assert policy.rerank_candidates == 20
    assert policy.final_top_k == 5


def test_retrieval_policy_resolves_top_k():
    policy = RetrievalPolicy(final_top_k=5)
    assert policy.resolve_top_k(0) == 5
    assert policy.resolve_top_k(3) == 3


def test_retrieval_policy_candidate_window():
    policy = RetrievalPolicy(rerank_candidates=20, final_top_k=5)
    assert policy.candidate_window(100) == 20
    assert policy.candidate_window(8) == 8
