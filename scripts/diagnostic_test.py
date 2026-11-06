#!/usr/bin/env python3
"""
Diagnostic script to test all API endpoints and identify issues
Run this to see what's working and what's not
"""

import asyncio
import aiohttp
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
API_BASE_URL = "http://localhost:7600"
FRONTEND_URL = "http://localhost:7500"
MCP_PORT = 7650

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_status(message: str, status: str = "info"):
    """Print colored status message"""
    if status == "success":
        print(f"{Colors.GREEN}✓{Colors.ENDC} {message}")
    elif status == "error":
        print(f"{Colors.RED}✗{Colors.ENDC} {message}")
    elif status == "warning":
        print(f"{Colors.YELLOW}⚠{Colors.ENDC} {message}")
    elif status == "info":
        print(f"{Colors.BLUE}ℹ{Colors.ENDC} {message}")
    else:
        print(f"  {message}")

async def test_endpoint(session: aiohttp.ClientSession, method: str, path: str, 
                       data: Optional[Dict] = None, description: str = "") -> bool:
    """Test a single endpoint"""
    url = f"{API_BASE_URL}{path}"
    try:
        async with session.request(method, url, json=data, timeout=5) as response:
            if response.status < 400:
                content = await response.text()
                try:
                    json_data = json.loads(content)
                    print_status(f"{method} {path}: {description or 'OK'}", "success")
                    return True
                except json.JSONDecodeError:
                    print_status(f"{method} {path}: Response is not JSON", "warning")
                    return True
            else:
                print_status(f"{method} {path}: HTTP {response.status}", "error")
                return False
    except asyncio.TimeoutError:
        print_status(f"{method} {path}: Timeout", "error")
        return False
    except aiohttp.ClientConnectionError:
        print_status(f"{method} {path}: Connection refused", "error")
        return False
    except Exception as e:
        print_status(f"{method} {path}: {str(e)}", "error")
        return False

async def check_backend_health():
    """Check if backend is running and healthy"""
    print(f"\n{Colors.BOLD}=== Checking Backend API ==={Colors.ENDC}")
    
    async with aiohttp.ClientSession() as session:
        # Test root endpoint
        if not await test_endpoint(session, "GET", "/", description="Root endpoint"):
            print_status("Backend may not be running on port 7600", "error")
            print_status("Start it with: python -m backend.main", "info")
            return False
        
        # Test health endpoint
        await test_endpoint(session, "GET", "/health", description="Health check")
        
        # Test API documentation
        await test_endpoint(session, "GET", "/api/docs", description="API docs (FastAPI)")
        
        return True

async def check_api_endpoints():
    """Test all API endpoints"""
    print(f"\n{Colors.BOLD}=== Testing API Endpoints ==={Colors.ENDC}")
    
    endpoints_to_test = [
        ("GET", "/api/endpoints", None, "List endpoints"),
        ("GET", "/api/cookbook", None, "Get cookbook recipes"),
        ("POST", "/api/chat", {"message": "Hello"}, "Chat endpoint"),
        ("GET", "/api/mcp/tools", None, "List MCP tools"),
        ("GET", "/api/openapi", None, "OpenAPI spec"),
    ]
    
    async with aiohttp.ClientSession() as session:
        results = []
        for method, path, data, desc in endpoints_to_test:
            result = await test_endpoint(session, method, path, data, desc)
            results.append(result)
        
        return all(results)

async def check_websocket():
    """Test WebSocket connection"""
    print(f"\n{Colors.BOLD}=== Testing WebSocket ==={Colors.ENDC}")
    
    ws_url = f"ws://localhost:7600/ws/chat"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url, timeout=5) as ws:
                # Send a test message
                await ws.send_str(json.dumps({"message": "test"}))
                
                # Wait for response
                msg = await asyncio.wait_for(ws.receive(), timeout=5)
                if msg.type == aiohttp.WSMsgType.TEXT:
                    print_status(f"WebSocket connected and responsive", "success")
                    return True
                else:
                    print_status(f"WebSocket unexpected response type", "warning")
                    return False
    except asyncio.TimeoutError:
        print_status(f"WebSocket connection timeout", "error")
        return False
    except Exception as e:
        print_status(f"WebSocket error: {str(e)}", "error")
        return False

async def check_frontend():
    """Check if frontend is accessible"""
    print(f"\n{Colors.BOLD}=== Checking Frontend ==={Colors.ENDC}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(FRONTEND_URL, timeout=5) as response:
                if response.status == 200:
                    print_status(f"Frontend accessible at {FRONTEND_URL}", "success")
                    return True
                else:
                    print_status(f"Frontend returned HTTP {response.status}", "error")
                    return False
    except aiohttp.ClientConnectionError:
        print_status(f"Frontend not running on port 7500", "error")
        print_status("Start it with: python -m http.server 7500 --directory frontend", "info")
        return False
    except Exception as e:
        print_status(f"Frontend error: {str(e)}", "error")
        return False

async def check_mcp_server():
    """Check if MCP server is running"""
    print(f"\n{Colors.BOLD}=== Checking MCP Server ==={Colors.ENDC}")
    
    try:
        # Check if MCP tools endpoint returns data
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{API_BASE_URL}/api/mcp/tools", timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    if "error" in data:
                        print_status(f"MCP is disabled: {data['error']}", "warning")
                        return False
                    elif "tools" in data:
                        tool_count = data.get("total", 0)
                        print_status(f"MCP server running with {tool_count} tools", "success")
                        return True
    except Exception as e:
        print_status(f"MCP check failed: {str(e)}", "error")
        return False

async def test_sample_requests():
    """Test some sample API requests"""
    print(f"\n{Colors.BOLD}=== Testing Sample Requests ==={Colors.ENDC}")
    
    sample_requests = [
        {
            "endpoint": "/api/chat",
            "method": "POST",
            "data": {
                "message": "What is the GetAccountBalance endpoint?",
                "context": []
            },
            "description": "Chat request about API"
        },
        {
            "endpoint": "/api/cookbook",
            "method": "GET",
            "data": None,
            "description": "Fetch cookbook recipes"
        }
    ]
    
    async with aiohttp.ClientSession() as session:
        for req in sample_requests:
            print(f"\n  Testing: {req['description']}")
            url = f"{API_BASE_URL}{req['endpoint']}"
            
            try:
                async with session.request(
                    req['method'], 
                    url, 
                    json=req['data'],
                    timeout=10
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        print_status(f"Response received", "success")
                        
                        # Show partial response
                        if isinstance(data, dict):
                            if 'response' in data:
                                preview = data['response'][:100] + "..." if len(data['response']) > 100 else data['response']
                                print(f"    Response preview: {preview}")
                        elif isinstance(data, list):
                            print(f"    Received {len(data)} items")
                    else:
                        error_text = await response.text()
                        print_status(f"HTTP {response.status}: {error_text[:100]}", "error")
            except Exception as e:
                print_status(f"Request failed: {str(e)}", "error")

async def check_file_structure():
    """Check if required files exist"""
    print(f"\n{Colors.BOLD}=== Checking File Structure ==={Colors.ENDC}")
    
    required_files = [
        "backend/main.py",
        "backend/config.py",
        "backend/knowledge_base.py",
        "backend/response_generator.py",
        "backend/apiweaver.py",
        "frontend/index.html",
        ".env"
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print_status(f"{file_path} exists", "success")
        else:
            print_status(f"{file_path} missing", "error")
            all_exist = False
    
    return all_exist

def check_env_file():
    """Check .env configuration"""
    print(f"\n{Colors.BOLD}=== Checking Environment Configuration ==={Colors.ENDC}")
    
    env_path = Path(".env")
    if not env_path.exists():
        print_status(".env file not found", "error")
        print_status("Create it with: cp .env.example .env", "info")
        return False
    
    with open(env_path) as f:
        env_content = f.read()
    
    required_vars = [
        "API_PORT",
        "FRONTEND_PORT",
        "MCP_ENABLED",
        "DEBUG"
    ]
    
    for var in required_vars:
        if var in env_content:
            print_status(f"{var} is configured", "success")
        else:
            print_status(f"{var} is missing", "warning")
    
    return True

async def main():
    """Run all diagnostic tests"""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("=" * 60)
    print("  TCS BaNCS API Cookbook - Diagnostic Tool")
    print("=" * 60)
    print(f"{Colors.ENDC}")
    
    # Check file structure first
    check_file_structure()
    check_env_file()
    
    # Check services
    backend_ok = await check_backend_health()
    
    if backend_ok:
        await check_api_endpoints()
        await check_websocket()
        await check_mcp_server()
        await test_sample_requests()
    
    await check_frontend()
    
    # Summary
    print(f"\n{Colors.BOLD}=== Summary ==={Colors.ENDC}")
    
    if not backend_ok:
        print_status("Backend is not running or not accessible", "error")
        print("\nTo fix:")
        print("1. Make sure you're in the project root directory")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Start the backend: python -m backend.main")
    else:
        print_status("Backend is running", "success")
    
    print(f"\n{Colors.BOLD}Quick Start Commands:{Colors.ENDC}")
    print("1. Start Backend:  python -m backend.main")
    print("2. Start Frontend: python -m http.server 7500 --directory frontend")
    print("3. Open Browser:   http://localhost:7500")

if __name__ == "__main__":
    asyncio.run(main())