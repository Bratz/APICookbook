# run_app.py (in project root)
"""
Run the TCS BAnCS API Cookbook application
Place this file in the project root directory
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    import uvicorn
    
    print("Starting TCS BAnCS API Cookbook...")
    print("API Server: http://localhost:7600")
    print("API Docs: http://localhost:7600/api/docs")
    print("Frontend: http://localhost:7600/")
    
    # Run using module import
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=7600,
        reload=False,
        log_level="info"
    )