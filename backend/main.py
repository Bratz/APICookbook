"""
TCS BAnCS API Cookbook - Complete Main Application with Hoppscotch Integration
Enhanced with improved error handling, validation, and fallback mechanisms
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import asyncio
import logging
import json
import os
import sys
import uuid
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING
from datetime import datetime, timezone
from pydantic import BaseModel, Field, validator
import httpx
import torch
import yaml
import traceback



# Fix import paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Create logs directory
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configure logging with both console and file output
def setup_logging():
    """Enhanced logging setup with proper debug capture"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set root logger to DEBUG to capture everything
    root_logger.setLevel(logging.DEBUG)
    
    # Console handler (INFO level for cleaner console)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)
    
    # Main log file (DEBUG level - captures everything)
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(file_handler)
    
    # Detailed debug log for response tracking
    debug_handler = RotatingFileHandler(
        'logs/debug_responses.log',
        maxBytes=10*1024*1024,
        backupCount=3
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(message)s'
    ))
    root_logger.addHandler(debug_handler)
    
    # Error log
    error_handler = RotatingFileHandler(
        'logs/error.log',
        maxBytes=10*1024*1024,
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(error_handler)

# Call this before creating the app
setup_logging()
logger = logging.getLogger(__name__)

try:
    from backend.bert_chatbot.integration import BERTChatbotIntegration, router as bert_router
    from backend.bert_chatbot.model import BERTCookbookChatbot
    BERT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"BERT components not available: {e}")
    BERT_AVAILABLE = False
    bert_router = None

# ============= BASE MODELS (Always defined) =============
class ChatMessageBase(BaseModel):
    """Base chat message model"""
    message: str
    context: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = Field(default_factory=list)
    
    def get_context_list(self) -> List[Dict[str, Any]]:
        """Ensure context is always a list"""
        if isinstance(self.context, dict):
            return [self.context]
        return self.context or []

class ChatResponseBase(BaseModel):
    """Base chat response model"""
    response: str
    code_examples: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    confidence: float = 0.8
    mcp_available: bool = False
    api_references: List[Dict[str, Any]] = Field(default_factory=list)

class CookbookRecipeBase(BaseModel):
    """Base cookbook recipe model"""
    id: str
    title: str
    description: str
    icon: str = "📚"
    tags: List[str] = Field(default_factory=list)
    content: Optional[str] = None
    category: str = "general"

# ============= FALLBACK CLASSES =============
class SimpleKnowledgeBase:
    """Minimal knowledge base implementation for fallback"""
    def __init__(self):
        self.endpoints = {
            "GetAccountBalance": {
                "method": "GET",
                "url": "/account/balance/{accountId}",
                "description": "Get account balance",
                "category": "account",
                "parameters": {
                    "accountId": {
                        "type": "string",
                        "required": True,
                        "description": "Account identifier"
                    }
                }
            },
            "GetCustomerDetails": {
                "method": "GET",
                "url": "/customer/{customerId}",
                "description": "Get customer details",
                "category": "customer",
                "parameters": {
                    "customerId": {
                        "type": "string",
                        "required": True,
                        "description": "Customer identifier"
                    }
                }
            }
        }
        self.mcp_tools = []
        self.http_client = None
    
    async def load_specifications(self):
        """Load basic endpoint specifications"""
        logger.info("Loaded fallback knowledge base with basic endpoints")
        return self.endpoints
    
    async def list_endpoints(self, category=None):
        """List endpoints, optionally filtered by category"""
        endpoints = []
        for name, endpoint in self.endpoints.items():
            if category and endpoint.get('category') != category:
                continue
            endpoints.append({
                "name": name,
                "method": endpoint.get("method", "GET"),
                "path": endpoint.get("url", "/"),
                "description": endpoint.get("description", ""),
                "category": endpoint.get("category", "general")
            })
        return endpoints
    
    async def get_cookbook_recipes(self, category=None):
        """Get cookbook recipes"""
        recipes = []
        for name, endpoint in self.endpoints.items():
            if category and endpoint.get('category') != category:
                continue
            recipes.append({
                "id": name.lower(),
                "title": name,
                "description": endpoint.get("description", ""),
                "category": endpoint.get("category", "general"),
                "icon": "📚",
                "tags": [endpoint.get("method", "GET"), endpoint.get("category", "general")]
            })
        return recipes
    
    async def get_recipe(self, recipe_id):
        """Get a specific recipe"""
        for name, endpoint in self.endpoints.items():
            if name.lower() == recipe_id:
                return {
                    "id": recipe_id,
                    "title": name,
                    "description": endpoint.get("description", ""),
                    "endpoint": endpoint
                }
        return None
    
    async def generate_openapi_spec(self, version="3.0"):
        """Generate basic OpenAPI spec"""
        spec = {
            "openapi": f"{version}.0",
            "info": {
                "title": "TCS BAnCS API",
                "version": "1.0.0",
                "description": "TCS BAnCS Banking API Documentation"
            },
            "servers": [
                {
                    "url": "https://demoapps.tcsbancs.com",
                    "description": "BAnCS API Server"
                }
            ],
            "paths": {}
        }
        
        for name, endpoint in self.endpoints.items():
            path = endpoint.get("url", "/")
            method = endpoint.get("method", "GET").lower()
            
            if path not in spec["paths"]:
                spec["paths"][path] = {}
            
            spec["paths"][path][method] = {
                "summary": endpoint.get("description", name),
                "operationId": name,
                "responses": {
                    "200": {"description": "Success"},
                    "400": {"description": "Bad Request"},
                    "500": {"description": "Internal Server Error"}
                }
            }
        
        return spec

class SimpleResponseGenerator:
    """Minimal response generator for fallback"""
    def __init__(self, kb):
        self.kb = kb
        self.default_responses = {
            "authentication": "Use headers: entity=GPRDTTSTOU, userId=1, languageCode=1",
            "endpoints": "Available endpoints: GetAccountBalance, GetCustomerDetails",
            "help": "I can help you with API endpoints, authentication, and code examples."
        }
    
    async def generate_response(self, message, context=None):
        """Generate a basic response"""
        message_lower = message.lower()
        
        # Check for keyword matches
        if "auth" in message_lower or "header" in message_lower:
            response = self.default_responses["authentication"]
        elif "endpoint" in message_lower or "api" in message_lower:
            response = self.default_responses["endpoints"]
        else:
            response = self.default_responses["help"]
        
        return {
            "response": response,
            "code_examples": [],
            "metadata": {"fallback_mode": True}
        }

class SimpleAPIWeaver:
    """Minimal API weaver for fallback"""
    def __init__(self):
        self.apis = {
            "default": {
                "name": "Default API",
                "endpoints": []
            }
        }
    
    async def _execute_api_call(self, api_name, endpoint_name, params, ctx):
        """Simulate API call execution"""
        return {
            "success": False,
            "message": "APIWeaver not available - running in fallback mode",
            "api": api_name,
            "endpoint": endpoint_name
        }

# ============= COMPONENT CHECKING =============
def check_required_components():
    """Check all required components are available"""
    components = {
        'config': False,
        'knowledge_base': False,
        'response_generator': False,
        'apiweaver': False,
        'models': False
    }
    
    imports = {}
    
    # Check config
    try:
        from backend.config import settings
        components['config'] = True
        imports['settings'] = settings
    except ImportError as e:
        logger.warning(f"Could not import config: {e}")
        # Create fallback settings
        class Settings:
            app_name = "TCS BAnCS API Cookbook"
            app_version = "1.0.0"
            api_port = 7600
            host = "0.0.0.0"
            debug = True
            mcp_enabled = True
            cors_origins = ["*"]
            cors_allow_credentials = True
            cors_allow_methods = ["*"]
            cors_allow_headers = ["*"]
            api_timeout = 30
            search_threshold = 60
            bancs_entity = "GPRDTTSTOU"
            bancs_user_id = "1"
            bancs_language_code = "1"
            api_base_url = "https://demoapps.tcsbancs.com"
            hoppscotch_port = 7550
            hoppscotch_enabled = True
            hoppscotch_self_hosted = True
        imports['settings'] = Settings()
    
    # Check models - use base models if import fails
    try:
        from backend.models import ChatMessage, ChatResponse, CookbookRecipe
        components['models'] = True
        imports['ChatMessage'] = ChatMessage
        imports['ChatResponse'] = ChatResponse
        imports['CookbookRecipe'] = CookbookRecipe
    except ImportError:
        logger.warning("Could not import models, using base models")
        imports['ChatMessage'] = ChatMessageBase
        imports['ChatResponse'] = ChatResponseBase
        imports['CookbookRecipe'] = CookbookRecipeBase
    
    # Check knowledge base
    try:
        from backend.knowledge_base import UnifiedAPIKnowledgeBase
        components['knowledge_base'] = True
        imports['UnifiedAPIKnowledgeBase'] = UnifiedAPIKnowledgeBase
    except ImportError:
        logger.warning("Could not import knowledge_base, using fallback")
        imports['UnifiedAPIKnowledgeBase'] = SimpleKnowledgeBase
    
    # Check response generator
    try:
        from backend.response_generator import UnifiedResponseGenerator
        components['response_generator'] = True
        imports['UnifiedResponseGenerator'] = UnifiedResponseGenerator
    except ImportError:
        logger.warning("Could not import response_generator, using fallback")
        imports['UnifiedResponseGenerator'] = SimpleResponseGenerator
    
    # Check APIWeaver
    try:
        from backend.apiweaver import BAnCSAPIWeaver
        components['apiweaver'] = True
        imports['BAnCSAPIWeaver'] = BAnCSAPIWeaver
    except ImportError:
        logger.warning("Could not import apiweaver, using fallback")
        imports['BAnCSAPIWeaver'] = SimpleAPIWeaver
    
    # Log component status
    logger.info(f"Component status: {components}")
    
    return components, imports

# Initialize components
components_loaded, imports = check_required_components()

# Extract imports
settings = imports['settings']
ChatMessage = imports['ChatMessage']
ChatResponse = imports['ChatResponse']
CookbookRecipe = imports['CookbookRecipe']
UnifiedAPIKnowledgeBase = imports['UnifiedAPIKnowledgeBase']
UnifiedResponseGenerator = imports['UnifiedResponseGenerator']
BAnCSAPIWeaver = imports['BAnCSAPIWeaver']

# ============= ENHANCED MODELS =============
class PlaygroundRequest(BaseModel):
    """Request model for playground execution"""
    type: str = Field(description="Type: 'mcp' or 'api'")
    # For MCP
    tool_name: Optional[str] = None
    tool_params: Optional[Dict[str, Any]] = None
    # For API
    method: Optional[str] = "GET"
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, Any]] = None
    body: Optional[Union[Dict[str, Any], str]] = None
    # Common
    use_auth: bool = True
    timeout: int = 30
    
    @validator('type')
    def validate_type(cls, v):
        if v not in ['mcp', 'api']:
            raise ValueError(f"Invalid type: {v}. Must be 'mcp' or 'api'")
        return v
    
    @validator('url')
    def validate_url_for_api(cls, v, values):
        if values.get('type') == 'api' and not v:
            raise ValueError("URL is required for API requests")
        return v
    
    @validator('tool_name')
    def validate_tool_for_mcp(cls, v, values):
        if values.get('type') == 'mcp' and not v:
            raise ValueError("Tool name is required for MCP requests")
        return v

class HoppscotchCollection(BaseModel):
    """Hoppscotch collection model"""
    v: int = 1
    name: str
    folders: List[Dict[str, Any]] = []
    requests: List[Dict[str, Any]] = []

# ============= ENHANCED LIFESPAN MANAGEMENT =============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management with enhanced error handling"""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Components loaded: {components_loaded}")
    
    # Initialize with defaults even if components fail
    app.state.kb = None
    app.state.response_generator = None
    app.state.apiweaver = None
    app.state.playground_history = []
    app.state.playground_sessions = {}
    app.state.is_fallback_mode = False
    
    try:
        # Initialize knowledge base
        try:
            app.state.kb = UnifiedAPIKnowledgeBase()
            await app.state.kb.load_specifications()
            logger.info("Knowledge base initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize knowledge base: {e}, using fallback")
            app.state.kb = SimpleKnowledgeBase()
            await app.state.kb.load_specifications()
            app.state.is_fallback_mode = True
        
        # Initialize response generator - MOVED BEFORE YIELD
        try:
            app.state.response_generator = UnifiedResponseGenerator(app.state.kb)
            logger.info("Response generator initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize response generator: {e}, using fallback")
            app.state.response_generator = SimpleResponseGenerator(app.state.kb)
        
        # Initialize BERT if available
        if BERT_AVAILABLE:
            try:
                app.state.bert_integration = BERTChatbotIntegration(app.state.kb)
                logger.info("BERT chatbot initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize BERT chatbot: {e}")
                app.state.bert_integration = None
        else:
            app.state.bert_integration = None
        
        # Initialize MCP if enabled
        if settings.mcp_enabled:
            try:
                app.state.apiweaver = BAnCSAPIWeaver()
                logger.info("APIWeaver (MCP) initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize APIWeaver: {e}, using fallback")
                app.state.apiweaver = SimpleAPIWeaver()
        else:
            logger.info("MCP is disabled in configuration")
            app.state.apiweaver = None
        
        # Initialize playground state
        app.state.playground_history = []
        app.state.playground_sessions = {}
        
        logger.info(f"API Server ready on port {settings.api_port}")
        logger.info(f"Fallback mode: {app.state.is_fallback_mode}")
        
    except Exception as e:
        logger.error(f"Critical error during initialization: {e}", exc_info=True)
        # Continue with degraded functionality rather than failing
        app.state.is_fallback_mode = True
        
        # Ensure we have at least fallback components
        if app.state.response_generator is None:
            app.state.response_generator = SimpleResponseGenerator(
                app.state.kb if app.state.kb else SimpleKnowledgeBase()
            )
    
    yield  # App runs here
    
    # Cleanup code after yield
    logger.info("Shutting down...")
    cleanup_tasks = []
    
    # Cleanup HTTP clients
    if hasattr(app.state, 'kb') and hasattr(app.state.kb, 'http_client') and app.state.kb.http_client:
        cleanup_tasks.append(app.state.kb.http_client.aclose())
    
    # Cleanup APIWeaver HTTP clients
    if hasattr(app.state, 'apiweaver') and hasattr(app.state.apiweaver, 'http_clients'):
        for client in app.state.apiweaver.http_clients.values():
            cleanup_tasks.append(client.aclose())
    
    if cleanup_tasks:
        results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Cleanup task {i} failed: {result}")
    
    logger.info("Shutdown complete")

# ============= FASTAPI APPLICATION =============
app = FastAPI(
    title=settings.app_name,
    description="Interactive API Cookbook with MCP tools and Hoppscotch integration",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

if BERT_AVAILABLE and bert_router:
    app.include_router(bert_router)
    logger.info("BERT chatbot endpoints mounted")

# ============= MIDDLEWARE =============
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Process time header middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add process time header to responses"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(f"{process_time:.3f}")
    return response

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        logger.info(f"[{request_id}] Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"[{request_id}] Request failed: {e}", exc_info=True)
        raise

# ============= STATIC FILES =============
# Mount frontend with validation
frontend_path = Path("frontend")
if frontend_path.exists():
    if (frontend_path / "index.html").exists():
        app.mount("/static", StaticFiles(directory="frontend", html=True), name="frontend")
        logger.info(f"Frontend mounted successfully from {frontend_path.absolute()}")
    else:
        logger.warning(f"Frontend directory exists but index.html not found at {frontend_path / 'index.html'}")
else:
    logger.warning(f"Frontend directory not found at {frontend_path.absolute()}")

# ============= CORE ENDPOINTS =============
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the cookbook UI"""
    frontend_file = Path("frontend/index.html")
    if frontend_file.exists():
        return FileResponse(frontend_file)
    
    # Fallback HTML if frontend not found
    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{settings.app_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
            h1 {{ color: #333; }}
            .status {{ padding: 10px; background: #e8f5e9; border-radius: 4px; margin: 10px 0; }}
            .links {{ margin-top: 20px; }}
            .links a {{ display: inline-block; margin: 5px 10px 5px 0; padding: 8px 16px; background: #2196F3; color: white; text-decoration: none; border-radius: 4px; }}
            .links a:hover {{ background: #1976D2; }}
            .warning {{ background: #fff3e0; padding: 10px; border-radius: 4px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{settings.app_name}</h1>
            <div class="status">✅ API Server is running!</div>
            {"<div class='warning'>⚠️ Running in fallback mode - some features may be limited</div>" if app.state.is_fallback_mode else ""}
            <p>Version: {settings.app_version}</p>
            <div class="links">
                <a href="/api/docs">📚 API Documentation</a>
                <a href="/health">❤️ Health Check</a>
                <a href="/api/endpoints">🔌 API Endpoints</a>
            </div>
            <h3>Quick Start:</h3>
            <ol>
                <li>Check the <a href="/api/docs">API Documentation</a> for available endpoints</li>
                <li>Use the <a href="/health">Health Check</a> to verify service status</li>
                <li>View available <a href="/api/endpoints">API Endpoints</a></li>
            </ol>
        </div>
    </body>
    </html>
    """)

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": settings.app_version,
        "fallback_mode": app.state.is_fallback_mode,
        "services": {
            "api": "running",
            "mcp": "disabled" if not settings.mcp_enabled else (
                "running" if app.state.apiweaver and not isinstance(app.state.apiweaver, SimpleAPIWeaver) else "fallback"
            ),
            "hoppscotch": "enabled" if settings.hoppscotch_enabled else "disabled",
            "knowledge_base": "fallback" if isinstance(app.state.kb, SimpleKnowledgeBase) else "running",
            "response_generator": "fallback" if isinstance(app.state.response_generator, SimpleResponseGenerator) else "running"
        },
        "components": components_loaded,
        "stats": {
            "endpoints_loaded": len(app.state.kb.endpoints) if app.state.kb else 0,
            "mcp_tools": len(getattr(app.state.kb, 'mcp_tools', [])),
            "playground_history": len(app.state.playground_history)
        }
    }
    
    # Determine overall health
    if app.state.is_fallback_mode:
        health_status["status"] = "degraded"
        health_status["message"] = "Running with limited functionality in fallback mode"
    
    return health_status

# ============= KNOWLEDGE BASE ENDPOINTS =============
@app.post("/api/knowledge/search")
async def search_knowledge(request: Request):
    """Search the knowledge base for relevant information"""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Knowledge base search request")
    
    try:
        data = await request.json()
        query = data.get('query', '')
        limit = data.get('limit', 5)
        
        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "Query parameter is required"}
            )
        
        logger.info(f"[{request_id}] Searching for: {query}")
        
        # Try BERT-based search first if available
        if hasattr(app.state, 'bert_integration') and app.state.bert_integration:
            try:
                # Use BERT to find relevant APIs
                bert_chatbot = app.state.bert_integration.bert_chatbot
                if bert_chatbot and hasattr(bert_chatbot, 'api_embeddings'):
                    # Generate embeddings for the query
                    query_embeddings = bert_chatbot._generate_embeddings(query)
                    
                    results = []
                    for endpoint_key, api_embeddings in bert_chatbot.api_embeddings.items():
                        # Calculate similarity
                        similarity = torch.cosine_similarity(query_embeddings, api_embeddings, dim=0)
                        score = similarity.item()
                        
                        if score > 0.3:  # Threshold for relevance
                            metadata = bert_chatbot.api_metadata.get(endpoint_key, {})
                            
                            # Parse endpoint key
                            parts = endpoint_key.split(' ', 1)
                            method = parts[0] if len(parts) > 0 else 'GET'
                            path = parts[1] if len(parts) > 1 else endpoint_key
                            
                            results.append({
                                'endpoint_name': endpoint_key,
                                'method': method,
                                'path': path,
                                'score': score,
                                'confidence': score,
                                'description': metadata.get('description', ''),
                                'parameters': metadata.get('parameters', []),
                                'title': endpoint_key
                            })
                    
                    # Sort by score
                    results.sort(key=lambda x: x['score'], reverse=True)
                    results = results[:limit]
                    
                    if results:
                        logger.info(f"[{request_id}] Found {len(results)} results via BERT")
                        return JSONResponse(content={
                            "status": "success",
                            "query": query,
                            "results": results,
                            "total": len(results),
                            "source": "bert"
                        })
            except Exception as e:
                logger.error(f"[{request_id}] BERT search error: {e}")
        
        # Fallback to keyword-based search
        logger.info(f"[{request_id}] Using keyword-based search")
        results = []
        query_lower = query.lower()
        query_words = query_lower.split()
        
        # Search in knowledge base endpoints
        if hasattr(app.state, 'kb') and app.state.kb:
            for endpoint_name, endpoint_data in app.state.kb.endpoints.items():
                # Calculate relevance score
                score = 0
                endpoint_lower = endpoint_name.lower()
                
                # Exact match in endpoint name
                if query_lower in endpoint_lower:
                    score += 0.8
                
                # Check description
                description = endpoint_data.get('description', '').lower()
                if query_lower in description:
                    score += 0.5
                
                # Check for word matches
                for word in query_words:
                    if len(word) > 2:  # Skip very short words
                        if word in endpoint_lower:
                            score += 0.3
                        if word in description:
                            score += 0.2
                        
                        # Check in URL/path
                        url = endpoint_data.get('url', '').lower()
                        if word in url:
                            score += 0.2
                        
                        # Check in parameters
                        params = endpoint_data.get('parameters', {})
                        if isinstance(params, dict):
                            for param_name in params.keys():
                                if word in param_name.lower():
                                    score += 0.1
                
                if score > 0:
                    results.append({
                        'endpoint_name': endpoint_name,
                        'method': endpoint_data.get('method', 'GET'),
                        'path': endpoint_data.get('url', '/'),
                        'score': min(score, 1.0),
                        'confidence': min(score, 1.0),
                        'description': endpoint_data.get('description', ''),
                        'parameters': endpoint_data.get('parameters', {}),
                        'title': endpoint_name,
                        'category': endpoint_data.get('category', 'general')
                    })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:limit]
        
        logger.info(f"[{request_id}] Found {len(results)} results in keyword search")
        
        return JSONResponse(content={
            "status": "success",
            "query": query,
            "results": results,
            "total": len(results),
            "source": "keyword"
        })
        
    except Exception as e:
        logger.error(f"[{request_id}] Knowledge search error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Knowledge search failed: {str(e)}"}
        )

@app.post("/api/knowledge/add")
async def add_to_knowledge(request: Request):
    """Add new information to the knowledge base"""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Add to knowledge base request")
    
    try:
        data = await request.json()
        endpoint_name = data.get('endpoint_name')
        endpoint_data = data.get('data', {})
        
        if not endpoint_name:
            return JSONResponse(
                status_code=400,
                content={"error": "endpoint_name is required"}
            )
        
        # Add to knowledge base
        if hasattr(app.state, 'kb') and app.state.kb:
            app.state.kb.endpoints[endpoint_name] = endpoint_data
            logger.info(f"[{request_id}] Added {endpoint_name} to knowledge base")
            
            # Update BERT embeddings if available
            if hasattr(app.state, 'bert_integration') and app.state.bert_integration:
                try:
                    bert_chatbot = app.state.bert_integration.bert_chatbot
                    if bert_chatbot:
                        description = f"{endpoint_data.get('description', '')} {endpoint_name}"
                        embeddings = bert_chatbot._generate_embeddings(description)
                        bert_chatbot.api_embeddings[endpoint_name] = embeddings
                        bert_chatbot.api_metadata[endpoint_name] = {
                            'method': endpoint_data.get('method', 'GET'),
                            'path': endpoint_data.get('url', '/'),
                            'parameters': endpoint_data.get('parameters', {}),
                            'description': endpoint_data.get('description', '')
                        }
                        logger.info(f"[{request_id}] Updated BERT embeddings for {endpoint_name}")
                except Exception as e:
                    logger.warning(f"[{request_id}] Failed to update BERT embeddings: {e}")
            
            return JSONResponse(content={
                "status": "success",
                "message": f"Added {endpoint_name} to knowledge base",
                "endpoint": endpoint_name
            })
        else:
            return JSONResponse(
                status_code=503,
                content={"error": "Knowledge base not available"}
            )
        
    except Exception as e:
        logger.error(f"[{request_id}] Add to knowledge error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to add to knowledge base: {str(e)}"}
        )

@app.get("/api/knowledge/endpoints")
async def get_knowledge_endpoints():
    """Get all endpoints in the knowledge base"""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Get knowledge endpoints request")
    
    try:
        endpoints = []
        
        if hasattr(app.state, 'kb') and app.state.kb:
            for name, data in app.state.kb.endpoints.items():
                endpoints.append({
                    'name': name,
                    'method': data.get('method', 'GET'),
                    'path': data.get('url', '/'),
                    'description': data.get('description', ''),
                    'parameters': data.get('parameters', {}),
                    'category': data.get('category', 'general')
                })
        
        return JSONResponse(content={
            "status": "success",
            "endpoints": endpoints,
            "total": len(endpoints)
        })
        
    except Exception as e:
        logger.error(f"[{request_id}] Get endpoints error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get endpoints: {str(e)}"}
        )

@app.delete("/api/knowledge/{endpoint_name}")
async def remove_from_knowledge(endpoint_name: str):
    """Remove an endpoint from the knowledge base"""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Remove from knowledge base: {endpoint_name}")
    
    try:
        if hasattr(app.state, 'kb') and app.state.kb:
            if endpoint_name in app.state.kb.endpoints:
                del app.state.kb.endpoints[endpoint_name]
                
                # Remove from BERT embeddings if available
                if hasattr(app.state, 'bert_integration') and app.state.bert_integration:
                    bert_chatbot = app.state.bert_integration.bert_chatbot
                    if bert_chatbot:
                        if endpoint_name in bert_chatbot.api_embeddings:
                            del bert_chatbot.api_embeddings[endpoint_name]
                        if endpoint_name in bert_chatbot.api_metadata:
                            del bert_chatbot.api_metadata[endpoint_name]
                
                logger.info(f"[{request_id}] Removed {endpoint_name} from knowledge base")
                return JSONResponse(content={
                    "status": "success",
                    "message": f"Removed {endpoint_name} from knowledge base"
                })
            else:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Endpoint {endpoint_name} not found"}
                )
        else:
            return JSONResponse(
                status_code=503,
                content={"error": "Knowledge base not available"}
            )
            
    except Exception as e:
        logger.error(f"[{request_id}] Remove from knowledge error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to remove from knowledge base: {str(e)}"}
        )

@app.get("/api/knowledge/search/{query}")
async def search_knowledge_get(query: str, limit: int = 5):
    """GET version of knowledge search for easier testing"""
    return await search_knowledge(Request({
        "type": "http",
        "method": "POST",
        "url": "/api/knowledge/search",
        "headers": {},
        "query_string": b"",
        "body": json.dumps({"query": query, "limit": limit}).encode()
    }, receive=lambda: {"type": "http.request", "body": json.dumps({"query": query, "limit": limit}).encode()}))



@app.post("/api/chat")
async def chat(message_data: Dict[str, Any]):
    """Enhanced chat endpoint with proper logging"""
    request_id = str(uuid.uuid4())[:8]
    
    try:
        # Parse the message
        if isinstance(message_data, dict):
            message_text = message_data.get('message', '')
            context = message_data.get('context', [])
        else:
            message_text = str(message_data)
            context = []
        
        # CRITICAL: Add comprehensive request logging
        logger.debug(f"[{request_id}] ========== CHAT REQUEST START ==========")
        logger.debug(f"[{request_id}] Message: {message_text}")
        logger.debug(f"[{request_id}] Context items: {len(context)}")
        logger.debug(f"[{request_id}] Full request data: {json.dumps(message_data, indent=2)}")
        
        if not message_text:
            logger.warning(f"[{request_id}] Empty message received")
            return {
                "response": "Please provide a message.",
                "confidence": 0,
                "mcp_available": False,
                "api_references": []
            }
        
        # Check for MCP-specific queries
        message_lower = message_text.lower()
        if 'mcp' in message_lower or 'tool' in message_lower:
            logger.info(f"[{request_id}] MCP-related query detected")
            mcp_response = await get_mcp_tools_response(app.state)
            if mcp_response:
                logger.debug(f"[{request_id}] MCP Response: {json.dumps(mcp_response, indent=2)}")
                logger.info(f"[{request_id}] ✓ Response source: MCP")
                return mcp_response
        
        # Try BERT chatbot if available
        if hasattr(app.state, 'bert_integration') and app.state.bert_integration:
            try:
                logger.info(f"[{request_id}] Attempting BERT integration...")
                bert_response = await app.state.bert_integration.process_developer_query(
                    query=message_text,
                    context={'messages': context} if context else None
                )
                
                # CRITICAL: Log the full BERT response
                logger.debug(f"[{request_id}] Raw BERT response: {json.dumps(bert_response, indent=2)}")
                
                response_text = bert_response.get('response') or bert_response.get('message', '')
                
                if response_text and bert_response.get('status') != 'error':
                    logger.info(f"[{request_id}] ✓ Response source: BERT")
                    logger.info(f"[{request_id}] BERT confidence: {bert_response.get('confidence', 'unknown')}")
                    logger.info(f"[{request_id}] BERT intent: {bert_response.get('intent', 'unknown')}")
                    logger.debug(f"[{request_id}] BERT response length: {len(response_text)} chars")
                    
                    # Add metadata
                    bert_response['metadata'] = {
                        'source': 'bert',
                        'request_id': request_id
                    }
                    
                    logger.debug(f"[{request_id}] Final BERT response: {json.dumps(bert_response, indent=2)}")
                    return bert_response
                else:
                    logger.warning(f"[{request_id}] BERT returned empty/error response")
                    logger.debug(f"[{request_id}] BERT error details: {json.dumps(bert_response, indent=2)}")
                    
            except Exception as e:
                logger.error(f"[{request_id}] BERT processing failed: {e}", exc_info=True)
        else:
            logger.debug(f"[{request_id}] BERT integration not available")
        
        # Fallback to response generator
        logger.info(f"[{request_id}] Using fallback response generator")
        
        if hasattr(app.state, 'response_generator'):
            try:
                response_data = await app.state.response_generator.generate_response(
                    message_text,
                    context=context
                )
                
                # CRITICAL: Log the fallback response
                logger.debug(f"[{request_id}] Fallback response: {json.dumps(response_data, indent=2)}")
                logger.info(f"[{request_id}] ✓ Response source: FALLBACK")
                
                response_data['metadata'] = {
                    'source': 'fallback',
                    'request_id': request_id
                }
                
                return response_data
                
            except Exception as e:
                logger.error(f"[{request_id}] Fallback generator failed: {e}", exc_info=True)
        
        # Ultimate fallback
        logger.warning(f"[{request_id}] Using hardcoded fallback response")
        logger.info(f"[{request_id}] ✓ Response source: HARDCODED")
        
        hardcoded_response = {
            "response": "I can help you with TCS BAnCS API endpoints...",
            "confidence": 0.3,
            "mcp_available": settings.mcp_enabled,
            "api_references": [],
            "metadata": {
                "source": "hardcoded",
                "request_id": request_id
            }
        }
        
        logger.debug(f"[{request_id}] Hardcoded response: {json.dumps(hardcoded_response, indent=2)}")
        return hardcoded_response
        
    except Exception as e:
        logger.error(f"[{request_id}] Chat endpoint error: {e}", exc_info=True)
        return {
            "response": f"Error: {str(e)}" if settings.debug else "An error occurred.",
            "confidence": 0,
            "mcp_available": False,
            "api_references": [],
            "metadata": {"error": True, "request_id": request_id}
        }
    finally:
        logger.debug(f"[{request_id}] ========== CHAT REQUEST END ==========")


async def get_mcp_tools_response(app_state) -> Optional[Dict[str, Any]]:
    """Generate response about MCP tools"""
    try:
        # Check if MCP is enabled
        if not settings.mcp_enabled:
            return {
                "response": "MCP (Model Context Protocol) tools are currently disabled. Enable MCP in the configuration to use these tools.",
                "confidence": 1.0,
                "mcp_available": False,
                "api_references": []
            }
        
        # Get MCP tools from knowledge base
        tools = []
        tool_descriptions = []
        
        if hasattr(app_state, 'kb') and hasattr(app_state.kb, 'mcp_tools'):
            for tool in app_state.kb.mcp_tools:
                tool_name = getattr(tool, 'name', 'unknown')
                tool_desc = getattr(tool, 'description', 'No description')
                tools.append(tool_name)
                tool_descriptions.append(f"• **{tool_name}**: {tool_desc}")
        
        # Also check APIWeaver
        if hasattr(app_state, 'apiweaver') and app_state.apiweaver:
            if hasattr(app_state.apiweaver, 'apis'):
                for api_name, api_config in app_state.apiweaver.apis.items():
                    if hasattr(api_config, 'endpoints'):
                        for endpoint in api_config.endpoints:
                            tool_name = f"{api_name}_{getattr(endpoint, 'name', 'unknown')}"
                            if tool_name not in tools:
                                tools.append(tool_name)
                                tool_descriptions.append(f"• **{tool_name}**: {getattr(endpoint, 'description', 'API endpoint')}")
        
        if tools:
            response = f"**Available MCP Tools ({len(tools)} total):**\n\n"
            response += "\n".join(tool_descriptions[:10])  # Show first 10
            
            if len(tools) > 10:
                response += f"\n\n...and {len(tools) - 10} more tools."
            
            response += "\n\n**How to use MCP tools:**\n"
            response += "These tools allow AI assistants to directly interact with the BAnCS API endpoints. "
            response += "Each tool corresponds to an API endpoint and can be called with the appropriate parameters.\n\n"
            response += "**Common tools include:**\n"
            response += "• Account management (balance, details, transactions)\n"
            response += "• Customer management (create, update, search)\n"
            response += "• Loan operations (details, repayment schedules)\n"
            response += "• Transaction processing (bookings, transfers)"
            
            return {
                "response": response,
                "confidence": 1.0,
                "mcp_available": True,
                "api_references": [],
                "metadata": {
                    "total_tools": len(tools),
                    "tools_list": tools[:20]  # Include first 20 in metadata
                }
            }
        else:
            return {
                "response": "MCP tools are enabled but no tools are currently loaded. This might be because:\n" +
                           "• API specifications haven't been loaded yet\n" +
                           "• The MCP server is still initializing\n" +
                           "• There's an issue with the API configuration\n\n" +
                           "Try restarting the server or check the logs for more information.",
                "confidence": 0.8,
                "mcp_available": True,
                "api_references": []
            }
            
    except Exception as e:
        logger.error(f"Error getting MCP tools: {e}")
        return {
            "response": "I encountered an error while fetching MCP tools. Please check the server logs.",
            "confidence": 0.5,
            "mcp_available": False,
            "api_references": [],
            "metadata": {"error": str(e)}
        }


@app.get("/api/cookbook")
async def get_cookbook_recipes(category: Optional[str] = None):
    """Get cookbook recipes with error handling"""
    try:
        recipes = await app.state.kb.get_cookbook_recipes(category)
        return recipes or []
    except Exception as e:
        logger.error(f"Cookbook error: {e}")
        return []

@app.get("/api/cookbook/{recipe_id}")
async def get_cookbook_recipe(recipe_id: str):
    """Get specific cookbook recipe"""
    try:
        recipe = await app.state.kb.get_recipe(recipe_id)
        if recipe:
            return recipe
        raise HTTPException(status_code=404, detail=f"Recipe '{recipe_id}' not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recipe fetch error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch recipe")

@app.get("/api/endpoints")
async def list_endpoints(category: Optional[str] = None):
    """List all API endpoints with enhanced details"""
    try:
        endpoints = await app.state.kb.list_endpoints(category)
        return endpoints or []
    except Exception as e:
        logger.error(f"Endpoints error: {e}")
        return []

@app.get("/api/openapi")
async def get_openapi_spec(version: str = "3.0"):
    """Generate OpenAPI specification"""
    try:
        if hasattr(app.state.kb, 'generate_openapi_spec'):
            spec = await app.state.kb.generate_openapi_spec(version)
            return spec
        else:
            # Fallback: generate basic spec
            return await app.state.kb.generate_openapi_spec(version)
    except Exception as e:
        logger.error(f"OpenAPI generation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate OpenAPI specification")

# ============= MCP ENDPOINTS =============
@app.get("/api/mcp/tools")
async def list_mcp_tools():
    """List available MCP tools with enhanced information"""
    if not settings.mcp_enabled:
        return {
            "error": "MCP is disabled",
            "tools": [],
            "total": 0,
            "enabled": False
        }
    
    try:
        tools = []
        
        if hasattr(app.state, 'apiweaver') and app.state.apiweaver:
            if hasattr(app.state.apiweaver, 'apis'):
                for api_name, api_config in app.state.apiweaver.apis.items():
                    if hasattr(api_config, 'endpoints'):
                        for endpoint in api_config.endpoints:
                            tools.append({
                                "name": f"{api_name}_{endpoint.name}",
                                "description": endpoint.description if hasattr(endpoint, 'description') else "",
                                "method": endpoint.method if hasattr(endpoint, 'method') else "POST",
                                "path": endpoint.path if hasattr(endpoint, 'path') else "/",
                                "api": api_name
                            })
            elif hasattr(app.state.kb, 'mcp_tools'):
                # Fallback to knowledge base MCP tools
                for tool in app.state.kb.mcp_tools:
                    tools.append({
                        "name": tool.name if hasattr(tool, 'name') else str(tool),
                        "description": tool.description if hasattr(tool, 'description') else "",
                        "method": "POST",
                        "path": "/mcp/tool"
                    })
        
        return {
            "tools": tools,
            "total": len(tools),
            "enabled": True,
            "fallback": isinstance(app.state.apiweaver, SimpleAPIWeaver)
        }
        
    except Exception as e:
        logger.error(f"MCP tools error: {e}")
        return {
            "tools": [],
            "total": 0,
            "error": str(e),
            "enabled": settings.mcp_enabled
        }

# ============= ENHANCED PLAYGROUND ENDPOINTS =============
@app.post("/api/playground/execute")
async def execute_playground_request(request: PlaygroundRequest):
    """Execute playground request with enhanced error handling"""
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    
    logger.info(f"[{request_id}] Playground request: {request.type}")
    
    try:
        result = None
        
        if request.type == "mcp":
            # Execute MCP tool
            if not settings.mcp_enabled or not hasattr(app.state, 'apiweaver'):
                raise HTTPException(status_code=503, detail="MCP is not available")
            
            # FIX: Normalize tool name format
            tool_name = request.tool_name
            logger.info(f"[{request_id}] Original tool name: {tool_name}")
            
            # Replace dots with underscores to normalize format
            tool_name = tool_name.replace('.', '_')
            
            # Default values
            api_name = "TCS_BAnCS"  # Default API name
            endpoint_name = None
            
            # Try to extract endpoint name from various formats
            prefixes_to_remove = [
                "TCS_BaNCS_",
                "TCS_BAnCS_", 
                "TCS_BANCS_",
                "BAnCS_",
                "BANCS_"
            ]
            
            temp_endpoint = tool_name
            for prefix in prefixes_to_remove:
                if temp_endpoint.upper().startswith(prefix.upper()):
                    temp_endpoint = temp_endpoint[len(prefix):]
                    break
            
            # Clean up common API documentation prefixes
            cleanup_patterns = [
                "RestFul_API_Documentation_",
                "REST_API_",
                "API_"
            ]
            for pattern in cleanup_patterns:
                temp_endpoint = temp_endpoint.replace(pattern, "")
            
            # Check if API exists
            if api_name not in app.state.apiweaver.apis:
                # Try to find any matching API
                for reg_api_name in app.state.apiweaver.apis.keys():
                    if api_name.upper() in reg_api_name.upper() or reg_api_name.upper() in api_name.upper():
                        api_name = reg_api_name
                        break
                else:
                    available_apis = list(app.state.apiweaver.apis.keys())
                    logger.error(f"[{request_id}] API '{api_name}' not found. Available APIs: {available_apis}")
                    raise HTTPException(
                        status_code=404, 
                        detail=f"API '{api_name}' not found. Available: {', '.join(available_apis)}"
                    )
            
            # Get the API configuration
            api_config = app.state.apiweaver.apis.get(api_name)
            if not api_config:
                raise HTTPException(status_code=404, detail=f"API configuration for '{api_name}' not found")
            
            # Try to find matching endpoint
            endpoint_name = None
            if hasattr(api_config, 'endpoints'):
                # Try exact match first
                for endpoint in api_config.endpoints:
                    if endpoint.name == temp_endpoint:
                        endpoint_name = endpoint.name
                        break
                
                # Try case-insensitive match
                if not endpoint_name:
                    for endpoint in api_config.endpoints:
                        if endpoint.name.lower() == temp_endpoint.lower():
                            endpoint_name = endpoint.name
                            break
                
                # Try partial match
                if not endpoint_name:
                    for endpoint in api_config.endpoints:
                        # Check various matching patterns
                        endpoint_lower = endpoint.name.lower()
                        temp_lower = temp_endpoint.lower()
                        
                        if (temp_lower in endpoint_lower or 
                            endpoint_lower in temp_lower or
                            # Try matching without common words
                            temp_lower.replace('get', '').replace('create', '').replace('update', '').replace('delete', '') 
                            in endpoint_lower.lower()):
                            endpoint_name = endpoint.name
                            break
                
                # If still not found, log available endpoints
                if not endpoint_name:
                    available = [ep.name for ep in api_config.endpoints]
                    logger.error(f"[{request_id}] Endpoint '{temp_endpoint}' not found in API '{api_name}'")
                    logger.info(f"[{request_id}] Available endpoints: {available}")
                    
                    # Try to suggest closest match
                    from difflib import get_close_matches
                    suggestions = get_close_matches(temp_endpoint, available, n=3, cutoff=0.4)
                    
                    error_msg = f"Endpoint '{temp_endpoint}' not found in API '{api_name}'."
                    if suggestions:
                        error_msg += f" Did you mean: {', '.join(suggestions)}?"
                    else:
                        error_msg += f" Available endpoints: {', '.join(available[:5])}"
                    
                    raise HTTPException(status_code=404, detail=error_msg)
            else:
                raise HTTPException(
                    status_code=500, 
                    detail=f"API configuration for '{api_name}' has no endpoints defined"
                )
            
            logger.info(f"[{request_id}] Executing MCP tool: {api_name}.{endpoint_name}")
            
            # Execute the API call
            try:
                result = await app.state.apiweaver._execute_api_call(
                    api_name=api_name,
                    endpoint_name=endpoint_name,
                    params=request.tool_params or {},
                    ctx=None
                )
            except ValueError as e:
                logger.error(f"[{request_id}] Error executing API call: {e}")
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"[{request_id}] Unexpected error executing API call: {e}")
                raise HTTPException(status_code=500, detail=f"Error executing API call: {str(e)}")
            
        elif request.type == "api":
            # Execute direct API call (your existing code for API type)
            async with httpx.AsyncClient(timeout=request.timeout) as client:
                # Build URL
                if request.url.startswith("http"):
                    url = request.url
                else:
                    url = f"{settings.api_base_url}{request.url}"
                
                logger.info(f"[{request_id}] Executing API call: {request.method} {url}")
                
                # Add auth headers if needed
                headers = request.headers or {}
                if request.use_auth:
                    headers.update({
                        "entity": settings.bancs_entity,
                        "userId": str(settings.bancs_user_id),
                        "languageCode": str(settings.bancs_language_code)
                    })
                
                # Make request
                response = await client.request(
                    method=request.method,
                    url=url,
                    headers=headers,
                    params=request.params if request.method == "GET" else None,
                    json=request.body if request.method != "GET" and isinstance(request.body, dict) else None,
                    content=request.body if request.method != "GET" and isinstance(request.body, str) else None
                )
                
                # Parse response
                content_type = response.headers.get("content-type", "")
                
                result = {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "content_type": content_type
                }
                
                if "application/json" in content_type:
                    result["body"] = response.json()
                else:
                    result["body"] = response.text
        
        else:
            raise HTTPException(status_code=400, detail=f"Invalid request type: {request.type}")
        
        # Prepare response
        duration = int((time.time() - start_time) * 1000)  # milliseconds
        
        # Add to history
        history_entry = {
            "id": request_id,
            "type": request.type,
            "timestamp": datetime.utcnow().isoformat(),
            "duration": duration,
            "request": {
                "tool_name": request.tool_name,
                "params": request.tool_params
            } if request.type == "mcp" else {
                "method": request.method,
                "url": request.url,
                "headers": request.headers,
                "body": request.body
            },
            "response": result,
            "success": True
        }
        
        app.state.playground_history.insert(0, history_entry)
        
        # Limit history size
        if len(app.state.playground_history) > 100:
            app.state.playground_history = app.state.playground_history[:100]
        
        logger.info(f"[{request_id}] Request completed in {duration}ms")
        
        return {
            "success": True,
            "result": result,
            "duration": duration,
            "request_id": request_id
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
        
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error in playground: {e}", exc_info=True)
        
        # Add error to history
        error_entry = {
            "id": request_id,
            "type": request.type,
            "timestamp": datetime.utcnow().isoformat(),
            "duration": int((time.time() - start_time) * 1000),
            "request": request.dict(),
            "error": str(e),
            "success": False
        }
        app.state.playground_history.insert(0, error_entry)
        
        return {
            "success": False,
            "error": str(e),
            "duration": int((time.time() - start_time) * 1000),
            "request_id": request_id
        }

@app.get("/api/playground/history")
async def get_playground_history(limit: int = 50):
    """Get playground execution history"""
    history = app.state.playground_history[:limit]
    return {
        "history": history,
        "total": len(app.state.playground_history),
        "limit": limit
    }

@app.delete("/api/playground/history")
async def clear_playground_history():
    """Clear playground history"""
    count = len(app.state.playground_history)
    app.state.playground_history = []
    return {
        "success": True,
        "cleared": count,
        "message": f"Cleared {count} history items"
    }

# ============= HOPPSCOTCH INTEGRATION =============
@app.get("/api/hoppscotch/collection")
async def get_hoppscotch_collection():
    """Generate Hoppscotch collection from API endpoints"""
    try:
        collection = {
            "v": 1,
            "name": "TCS BAnCS APIs",
            "folders": [],
            "requests": []
        }
        
        # Group endpoints by category
        categories = {}
        for name, endpoint in app.state.kb.endpoints.items():
            category = endpoint.get('category', 'general')
            if category not in categories:
                categories[category] = []
            categories[category].append((name, endpoint))
        
        # Create folders and requests
        for category, endpoints in categories.items():
            folder = {
                "v": 1,
                "name": category.title(),
                "folders": [],
                "requests": []
            }
            
            for endpoint_name, endpoint in endpoints:
                # Create Hoppscotch request
                request = {
                    "v": "1",
                    "name": endpoint_name,
                    "method": endpoint.get('method', 'GET'),
                    "endpoint": "<<baseUrl>>" + endpoint.get('url', '/'),
                    "headers": [
                        {"key": "entity", "value": "<<entity>>", "active": True},
                        {"key": "userId", "value": "<<userId>>", "active": True},
                        {"key": "languageCode", "value": "<<languageCode>>", "active": True}
                    ],
                    "params": [],
                    "body": {
                        "contentType": None,
                        "body": None
                    },
                    "auth": {
                        "authType": "none",
                        "authActive": False
                    }
                }
                
                # Add parameters
                if endpoint.get('parameters'):
                    for param_name, param_info in endpoint['parameters'].items():
                        if "{" + param_name + "}" in endpoint.get('url', ''):
                            # Path parameter - already in URL
                            continue
                        else:
                            # Query parameter
                            request["params"].append({
                                "key": param_name,
                                "value": param_info.get('example', ''),
                                "active": param_info.get('required', False)
                            })
                
                # Add body for POST/PUT
                if endpoint.get('method') in ['POST', 'PUT', 'PATCH']:
                    request["body"]["contentType"] = "application/json"
                    request["body"]["body"] = json.dumps(
                        endpoint.get('body_example', {"example": "data"}),
                        indent=2
                    )
                
                folder["requests"].append(request)
            
            collection["folders"].append(folder)
        
        return collection
        
    except Exception as e:
        logger.error(f"Error generating Hoppscotch collection: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate collection")

@app.get("/api/debug/knowledge-base")
async def debug_knowledge_base():
    """Debug endpoint to see what's loaded"""
    return {
        "total_endpoints": len(app.state.kb.endpoints),
        "endpoints": list(app.state.kb.endpoints.keys()),
        "categories": list(app.state.kb.api_categories.keys()) if hasattr(app.state.kb, 'api_categories') else [],
        "mcp_tools": len(app.state.kb.mcp_tools) if hasattr(app.state.kb, 'mcp_tools') else 0,
        "is_fallback_mode": app.state.is_fallback_mode if hasattr(app.state, 'is_fallback_mode') else False
    }

@app.get("/api/hoppscotch/environment")
async def get_hoppscotch_environment():
    """Generate Hoppscotch environment variables"""
    return {
        "v": 1,
        "name": "BAnCS Demo",
        "variables": [
            {"key": "baseUrl", "value": settings.api_base_url},
            {"key": "entity", "value": settings.bancs_entity},
            {"key": "userId", "value": settings.bancs_user_id},
            {"key": "languageCode", "value": settings.bancs_language_code}
        ]
    }

# ============= ENHANCED WEBSOCKET =============
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """Enhanced WebSocket endpoint with proper logging"""
    client_id = str(uuid.uuid4())[:8]
    await websocket.accept()
    logger.info(f"WebSocket client {client_id} connected")
    
    try:
        while True:
            try:
                # Receive message with timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=300)
                
                # CRITICAL: Add logging here!
                logger.debug(f"[WS-{client_id}] ========== WEBSOCKET MESSAGE START ==========")
                logger.debug(f"[WS-{client_id}] Raw data received: {data}")
                
                # Parse message
                try:
                    message_data = json.loads(data) if data else {}
                except json.JSONDecodeError:
                    message_data = {"message": data}
                
                logger.debug(f"[WS-{client_id}] Parsed message data: {json.dumps(message_data, indent=2)}")
                
                # Validate message
                if not message_data.get("message"):
                    logger.warning(f"[WS-{client_id}] Empty message received")
                    await websocket.send_json({
                        "error": "Message is required",
                        "type": "validation_error"
                    })
                    continue
                
                # Generate response
                try:
                    message_text = message_data.get("message", "")
                    context = message_data.get("context", [])
                    
                    logger.info(f"[WS-{client_id}] Processing message: {message_text[:100]}...")
                    
                    # Check for BERT first
                    response = None
                    response_source = "unknown"
                    
                    # Try BERT if available
                    if hasattr(app.state, 'bert_integration') and app.state.bert_integration:
                        try:
                            logger.info(f"[WS-{client_id}] Attempting BERT processing...")
                            bert_response = await app.state.bert_integration.process_developer_query(
                                query=message_text,
                                context={'messages': context} if context else None
                            )
                            
                            logger.debug(f"[WS-{client_id}] BERT response: {json.dumps(bert_response, indent=2)}")
                            
                            response_text = bert_response.get('response') or bert_response.get('message', '')
                            if response_text and bert_response.get('status') != 'error':
                                response = bert_response
                                response_source = "BERT"
                                logger.info(f"[WS-{client_id}] [SUCCESS] Using BERT response")
                            else:
                                logger.warning(f"[WS-{client_id}] BERT returned empty/error response")
                        except Exception as e:
                            logger.error(f"[WS-{client_id}] BERT failed: {e}")
                    
                    # Fallback to response generator if no BERT response
                    if not response and hasattr(app.state, 'response_generator'):
                        try:
                            logger.info(f"[WS-{client_id}] Using fallback response generator...")
                            response = await app.state.response_generator.generate_response(
                                message_text,
                                context=context
                            )
                            response_source = "FALLBACK"
                            logger.debug(f"[WS-{client_id}] Fallback response: {json.dumps(response, indent=2)}")
                            logger.info(f"[WS-{client_id}] [FALLBACK] Using fallback response")
                        except Exception as e:
                            logger.error(f"[WS-{client_id}] Fallback failed: {e}")
                    
                    # Ultimate fallback
                    if not response:
                        response = {
                            "response": "I can help you with API endpoints.",
                            "confidence": 0.3,
                            "mcp_available": False
                        }
                        response_source = "HARDCODED"
                        logger.warning(f"[WS-{client_id}] Using hardcoded fallback")
                    
                    # Add metadata
                    response["client_id"] = client_id
                    response["timestamp"] = datetime.now(timezone.utc).isoformat()
                    response["metadata"] = {
                        "source": response_source,
                        "websocket": True
                    }
                    
                    # LOG THE FINAL RESPONSE SOURCE
                    logger.info(f"[WS-{client_id}] ===== RESPONSE SOURCE: {response_source} =====")
                    logger.debug(f"[WS-{client_id}] Final response: {json.dumps(response, indent=2)}")
                    
                    await websocket.send_json(response)
                    

                    logger.debug(f"[WS-{client_id}] ========== WEBSOCKET MESSAGE END ==========")
                    
                except Exception as e:
                    logger.error(f"[WS-{client_id}] Error generating response: {e}", exc_info=True)
                    await websocket.send_json({
                        "error": "Failed to generate response",
                        "type": "generation_error",
                        "details": str(e) if settings.debug else None
                    })
                
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_json({"type": "ping"})
                
            except json.JSONDecodeError as e:
                logger.error(f"[WS-{client_id}] JSON decode error: {e}")
                await websocket.send_json({
                    "error": "Invalid JSON format",
                    "type": "parse_error",
                    "details": str(e) if settings.debug else None
                })
                
    except WebSocketDisconnect:
        logger.info(f"[WS-{client_id}] WebSocket disconnected normally")
        
    except Exception as e:
        logger.error(f"[WS-{client_id}] WebSocket error: {e}", exc_info=True)
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except:
            pass
    finally:
        logger.info(f"[WS-{client_id}] WebSocket connection closed")

# ============= MAIN ENTRY POINT =============
if __name__ == "__main__":
    import uvicorn
    
    # Determine if we're in fallback mode
    fallback_warning = (
        "\n⚠️  WARNING: Running in fallback mode with limited functionality!\n"
        "Some components could not be loaded. Check the logs for details.\n"
    ) if any(not v for v in components_loaded.values()) else ""
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║           TCS BAnCS API Cookbook v{settings.app_version}              ║
    ╚══════════════════════════════════════════════════════════════╝
    {fallback_warning}
    Starting server...
    - API Server: http://localhost:{settings.api_port}
    - API Docs: http://localhost:{settings.api_port}/api/docs
    - Health Check: http://localhost:{settings.api_port}/health
    
    Components Status:
    {json.dumps(components_loaded, indent=2)}
    """)
    
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info"
    )