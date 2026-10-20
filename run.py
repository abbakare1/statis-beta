"""STATIS entry point — run the FastAPI application."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn
from config.settings import get_settings

settings = get_settings()

if __name__ == "__main__":
    print(f"Starting STATIS on http://{settings.HOST}:{settings.PORT}")
    print(f"Ollama: {settings.OLLAMA_BASE_URL} ({settings.OLLAMA_MODEL})")
    uvicorn.run(
        "app.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
