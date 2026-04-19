"""
Pipeline singleton — imported by all route modules.
Instantiated once; ingest() is called from the FastAPI lifespan.
"""
import sys
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), ".env"))

# Ensure the project root is on sys.path so sl_rag can be imported
# when uvicorn is launched from the app/backend directory.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from sl_rag.pipeline import SLRAGPipeline  # noqa: E402

pipe = SLRAGPipeline(
    data_dir=os.path.join(_project_root, "data"),
    storage_dir=os.path.join(_project_root, "storage"),
    config_path=os.path.join(_project_root, "config", "config.yaml"),
    use_gpu=True,
    encryption=True,
    load_llm=True,
    llm_model=(
        os.getenv("LM_STUDIO_MODEL", "").strip()
        or os.getenv("OPENAI_MODEL", "").strip()
        or os.getenv("LLM_MODEL", "").strip()
        or "google/gemma-4-e4b"
    ),
)
