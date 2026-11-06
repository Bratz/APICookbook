# run_app.py (in project root)
"""
Run the TCS BaNCS API Cookbook application
Place this file in the project root directory
"""

import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    import uvicorn
    from backend.config import settings

    # Binds to 127.0.0.1 by default (no auth on any route); set HOST in .env to expose it.
    base = f"http://{settings.host}:{settings.api_port}"
    print("Starting TCS BaNCS API Cookbook...")
    print(f"API Server: {base}")
    print(f"API Docs: {base}/api/docs")
    print(f"Frontend: {base}/")

    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.api_port,
        reload=False,
        log_level="info"
    )
