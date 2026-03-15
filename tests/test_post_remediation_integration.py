import importlib
import sys
import types

import numpy as np

from sl_rag.core.schemas import Chunk, Document


def _install_dependency_stubs():
    # torch stub
    torch_mod = types.ModuleType("torch")
    torch_mod.float16 = "float16"
    torch_mod.no_grad = lambda: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda *a: False)
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        get_device_name=lambda _i=0: "stub-gpu",
        get_device_properties=lambda _i=0: types.SimpleNamespace(total_memory=0),
        memory_allocated=lambda _i=0: 0,
        empty_cache=lambda: None,
    )
    sys.modules["torch"] = torch_mod

    # sentence_transformers stub
    st_mod = types.ModuleType("sentence_transformers")

    class _SentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, **kwargs):
            if isinstance(texts, str):
                return np.array([1.0, 0.0], dtype=np.float32)
            return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)

        def get_sentence_embedding_dimension(self):
            return 2

    class _CrossEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, pairs, **kwargs):
            return np.array([0.5 for _ in pairs], dtype=np.float32)

    st_mod.SentenceTransformer = _SentenceTransformer
    st_mod.CrossEncoder = _CrossEncoder
    sys.modules["sentence_transformers"] = st_mod

    # rank_bm25 stub
    bm25_mod = types.ModuleType("rank_bm25")

    class _BM25Okapi:
        def __init__(self, corpus, **kwargs):
            self._n = len(corpus)

        def get_scores(self, _query):
            return np.array([1.0 for _ in range(self._n)], dtype=np.float32)

    bm25_mod.BM25Okapi = _BM25Okapi
    sys.modules["rank_bm25"] = bm25_mod

    # transformers stub
    tr_mod = types.ModuleType("transformers")

    class _BitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _Tokenizer:
        pad_token = None
        eos_token = ""
        pad_token_id = 0

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def __call__(self, text, **kwargs):
            class _T(dict):
                def to(self, _device):
                    return self

            return _T({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]})

        def decode(self, _tokens, **kwargs):
            return "ANSWER: stub"

    class _Model:
        device = "cpu"

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def eval(self):
            return None

        def generate(self, **kwargs):
            return [[1, 2, 3]]

    tr_mod.BitsAndBytesConfig = _BitsAndBytesConfig
    tr_mod.AutoTokenizer = _Tokenizer
    tr_mod.AutoModelForCausalLM = _Model
    sys.modules["transformers"] = tr_mod

    # faiss stub
    faiss_mod = types.ModuleType("faiss")

    class _BaseIndex:
        def __init__(self, d):
            self.d = d
            self.vectors = []
            self.ntotal = 0
            self.is_trained = True

        def add(self, vecs):
            self.vectors.extend(vecs.tolist())
            self.ntotal = len(self.vectors)

        def search(self, query, k):
            k = min(k, self.ntotal)
            distances = np.array([[1.0 for _ in range(k)]], dtype=np.float32)
            indices = np.array([[i for i in range(k)]], dtype=np.int64)
            return distances, indices

        def reconstruct(self, i):
            return np.array(self.vectors[i], dtype=np.float32)

        def train(self, _vecs):
            self.is_trained = True

    class _IndexFlatIP(_BaseIndex):
        pass

    class _IndexFlatL2(_BaseIndex):
        pass

    class _IndexIVFFlat(_BaseIndex):
        def __init__(self, quantizer, d, nlist, metric=None):
            super().__init__(d)

    faiss_mod.IndexFlat = _BaseIndex
    faiss_mod.IndexFlatIP = _IndexFlatIP
    faiss_mod.IndexFlatL2 = _IndexFlatL2
    faiss_mod.IndexIVFFlat = _IndexIVFFlat
    faiss_mod.METRIC_INNER_PRODUCT = 0
    faiss_mod.normalize_L2 = lambda _arr: None
    faiss_mod.serialize_index = lambda _idx: b"stub"
    faiss_mod.deserialize_index = lambda _bytes: _IndexFlatIP(2)
    sys.modules["faiss"] = faiss_mod


def _install_sklearn_stubs():
    sk_mod = types.ModuleType("sklearn")
    cluster_mod = types.ModuleType("sklearn.cluster")
    metrics_mod = types.ModuleType("sklearn.metrics")
    feat_mod = types.ModuleType("sklearn.feature_extraction")
    text_mod = types.ModuleType("sklearn.feature_extraction.text")

    class _KMeans:
        def __init__(self, n_clusters=2, **kwargs):
            self.n_clusters = n_clusters

        def fit_predict(self, x):
            return np.array([i % self.n_clusters for i in range(len(x))])

    class _TfidfVectorizer:
        def __init__(self, **kwargs):
            pass

        def fit_transform(self, texts):
            self._features = np.array(["finance", "security"])
            return np.array([[1.0, 0.5] for _ in texts])

        def get_feature_names_out(self):
            return self._features

    cluster_mod.KMeans = _KMeans
    metrics_mod.silhouette_score = lambda x, y: 0.2
    metrics_mod.davies_bouldin_score = lambda x, y: 1.0
    text_mod.TfidfVectorizer = _TfidfVectorizer

    sys.modules["sklearn"] = sk_mod
    sys.modules["sklearn.cluster"] = cluster_mod
    sys.modules["sklearn.metrics"] = metrics_mod
    sys.modules["sklearn.feature_extraction"] = feat_mod
    sys.modules["sklearn.feature_extraction.text"] = text_mod


def _import_pipeline_with_stubs():
    _install_dependency_stubs()
    _install_sklearn_stubs()
    return importlib.import_module("sl_rag.pipeline")


def test_ingest_and_query_integration_with_mocks(tmp_path, monkeypatch):
    pipeline_mod = _import_pipeline_with_stubs()

    class FakeLoader:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load_directory(self, d, recursive=False):
            doc = Document(
                doc_id="doc-1",
                content="Public doc text. Restricted annex text.",
                metadata={"filename": "mock.pdf", "filepath": "mock.pdf"},
            )
            return [doc], {"successful": 1}

    class FakeAnonymizer:
        def __init__(self, **kwargs):
            pass

        def anonymize(self, text):
            return text, {}

    class FakeChunker:
        def __init__(self, **kwargs):
            pass

        def chunk_document(self, doc):
            return [
                Chunk(
                    chunk_id="chunk-public",
                    doc_id=doc.doc_id,
                    content="Public info",
                    chunk_index=0,
                    start_char=0,
                    end_char=11,
                    token_count=2,
                    domain="finance",
                    metadata={"source_document": "mock.pdf", "sensitivity": "public"},
                ),
                Chunk(
                    chunk_id="chunk-restricted",
                    doc_id=doc.doc_id,
                    content="Restricted info",
                    chunk_index=1,
                    start_char=12,
                    end_char=27,
                    token_count=2,
                    domain="security",
                    metadata={"source_document": "mock.pdf", "sensitivity": "restricted"},
                ),
            ]

    class FakeEmbedder:
        def __init__(self, **kwargs):
            pass

        def generate_embeddings(self, chunks, normalize=True):
            for c in chunks:
                c.embedding = np.array([1.0, 0.0], dtype=np.float32)
            return chunks

        def generate_query_embedding(self, query, normalize=True):
            return np.array([1.0, 0.0], dtype=np.float32)

    class FakeFAISS:
        def __init__(self, *args, **kwargs):
            pass

        def add_chunks(self, chunks):
            return None

    class FakeDomainManager:
        def __init__(self, **kwargs):
            self.domains = {"finance": np.array([1.0, 0.0]), "security": np.array([0.0, 1.0])}

        def detect_domains(self, chunks):
            return {"num_domains": 2}

    class FakeBM25:
        def __init__(self, chunks):
            self.chunks = chunks

    class FakeHybrid:
        def __init__(self, bm25, faiss_idx, embedder, alpha=0.7):
            self._chunks = bm25.chunks

    class FakeReranker:
        def __init__(self, **kwargs):
            pass

    class FakeRetrievalPipeline:
        def __init__(self, embedder, domain_mgr, hybrid, reranker, **kwargs):
            self._chunks = hybrid._chunks

        def retrieve(self, q, top_k=5, enable_reranking=True):
            return [(c, 0.9 - (i * 0.1)) for i, c in enumerate(self._chunks)]

    class FakePromptBuilder:
        def __init__(self, **kwargs):
            pass

        def detect_injection(self, q):
            return False

        def build_prompt(self, q, r):
            return "prompt"

    class FakeValidation:
        def __init__(self, **kwargs):
            pass

        def validate_retrieval(self, q, results):
            return {"confidence": 0.8, "citation_quality": "good"}

        def validate_answer(self, ans, chunks, scores):
            return {"confidence": 0.7, "hallucination_risk": "low"}

    class SpyMonitor:
        def __init__(self, *args, **kwargs):
            self.logged_query = None
            self.logged_access = []
            self.pattern_called = False
            self.security_events = []
            self._drift_init = False

        def log_security_event(self, event_type, severity="info", details=""):
            self.security_events.append((event_type, severity, details))

        def check_domain_drift(self, domains, threshold=0.15, update_baseline=True):
            if not self._drift_init:
                self._drift_init = True
                return {"drift_detected": False, "mean_distance": 0.0, "domains_compared": 0}
            return {"drift_detected": False, "mean_distance": 0.0, "domains_compared": len(domains)}

        def log_query(self, query, domains, num_results, latency_ms, confidence=0.0, **kwargs):
            self.logged_query = {"query": query, "domains": domains, **kwargs}
            return 1

        def log_document_access(self, doc_id, chunk_ids, **kwargs):
            self.logged_access.append((doc_id, kwargs))

        def analyze_query_patterns(self, **kwargs):
            self.pattern_called = True
            return []

        def generate_compliance_report(self):
            return {"ok": True}

    monkeypatch.setattr(pipeline_mod, "DocumentLoader", FakeLoader)
    monkeypatch.setattr(pipeline_mod, "PIIAnonymizer", FakeAnonymizer)
    monkeypatch.setattr(pipeline_mod, "ChunkGenerator", FakeChunker)
    monkeypatch.setattr(pipeline_mod, "EmbeddingGenerator", FakeEmbedder)
    monkeypatch.setattr(pipeline_mod, "FAISSIndexManager", FakeFAISS)
    monkeypatch.setattr(pipeline_mod, "DocumentLevelDomainManager", FakeDomainManager)
    monkeypatch.setattr(pipeline_mod, "BM25Retriever", FakeBM25)
    monkeypatch.setattr(pipeline_mod, "HybridRetriever", FakeHybrid)
    monkeypatch.setattr(pipeline_mod, "CrossEncoderReranker", FakeReranker)
    monkeypatch.setattr(pipeline_mod, "RetrievalPipeline", FakeRetrievalPipeline)
    monkeypatch.setattr(pipeline_mod, "PromptBuilder", FakePromptBuilder)
    monkeypatch.setattr(pipeline_mod, "ValidationPipeline", FakeValidation)
    monkeypatch.setattr(pipeline_mod, "MonitoringSystem", SpyMonitor)

    pipe = pipeline_mod.SLRAGPipeline(
        data_dir=str(tmp_path),
        storage_dir=str(tmp_path / "storage"),
        load_llm=False,
    )

    ingest_result = pipe.ingest()
    assert "domains" in ingest_result
    assert "drift" in ingest_result

    allowed_resp = pipe.query(
        "show documents",
        generate_answer=False,
        auth_context={"user_id": "u-guest", "role": "guest", "session_id": "s-1"},
    )
    assert "error" not in allowed_resp
    # guest should only receive public sources
    assert len(allowed_resp["sources"]) == 1
    assert allowed_resp["sources"][0]["chunk_id"] == "chunk-public"
    assert pipe.monitor.logged_query["user_id"] == "u-guest"
    assert pipe.monitor.logged_query["role"] == "guest"
    assert pipe.monitor.pattern_called is True

    denied_resp = pipe.query(
        "show documents",
        generate_answer=False,
        auth_context={"user_id": "u-x", "role": "unknown_role"},
    )
    assert denied_resp["error"] == "Access denied for role"
    assert any(evt[0] == "query_access_denied" for evt in pipe.monitor.security_events)
