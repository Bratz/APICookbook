"""
TCS BaNCS API Cookbook - Complete Main Application with Hoppscotch Integration
Enhanced with improved error handling, validation, and fallback mechanisms
BERT/PyTorch are now completely optional
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
import yaml
import traceback
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Optional imports - PyTorch is NOT required
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Fix import paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

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
    
    # Fix for Windows encoding issues
    # Reconfigure in place: replacing sys.stdout orphans the original stream
    # (and breaks pytest's capture, which owns it during tests).
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding='utf-8', errors='replace')
    
    root_logger.addHandler(console_handler)
    
    # Main log file (DEBUG level - captures everything)
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(file_handler)
    
    # Error log
    error_handler = RotatingFileHandler(
        'logs/error.log',
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(error_handler)

# Call this before creating the app
setup_logging()
logger = logging.getLogger(__name__)

# ============= UTILITY FUNCTIONS =============
def sanitize_text(text: str) -> str:
    """Remove emojis and special characters that cause encoding issues"""
    if not text:
        return text
    
    # Replace emojis with text equivalents
    emoji_map = {
        '💡': '[TIP]', '🔗': '[LINK]', '✅': '[OK]', '❌': '[ERROR]',
        '📝': '[NOTE]', '🚀': '[START]', '⚠️': '[WARNING]', '📊': '[DATA]',
        '🔍': '[SEARCH]', '💻': '[CODE]', '📚': '[DOCS]', '❤️': '[HEALTH]',
        '🔌': '[API]', '📦': '[PACKAGE]', '🎯': '[TARGET]', '🔧': '[TOOL]',
        '➡️': '->', '⬅️': '<-', '✨': '*', '🌟': '*'
    }
    
    for emoji, replacement in emoji_map.items():
        text = text.replace(emoji, replacement)
    
    # On Windows, be more aggressive with cleaning
    if sys.platform == "win32":
        # Remove any remaining non-ASCII characters
        try:
            text = text.encode('ascii', 'ignore').decode('ascii')
        except:
            pass
    
    return text

def compute_similarity(vec1, vec2):
    """Compute similarity between two vectors (works with or without torch)"""
    try:
        if TORCH_AVAILABLE and torch is not None:
            if hasattr(vec1, '__class__') and 'torch' in str(type(vec1)):
                if hasattr(vec2, '__class__') and 'torch' in str(type(vec2)):
                    return torch.cosine_similarity(vec1, vec2, dim=0).item()
        
        # Handle sparse matrices from TF-IDF FIRST
        if hasattr(vec1, 'toarray'):
            vec1_np = vec1.toarray()
        elif hasattr(vec1, 'numpy'):
            vec1_np = vec1.numpy()
        else:
            vec1_np = np.asarray(vec1)  # Use asarray instead of array
        
        if hasattr(vec2, 'toarray'):
            vec2_np = vec2.toarray()
        elif hasattr(vec2, 'numpy'):
            vec2_np = vec2.numpy()
        else:
            vec2_np = np.asarray(vec2)  # Use asarray instead of array
        
        # Ensure proper shape
        if vec1_np.ndim == 1:
            vec1_np = vec1_np.reshape(1, -1)
        if vec2_np.ndim == 1:
            vec2_np = vec2_np.reshape(1, -1)
        
        # Check for dimension compatibility
        if vec1_np.shape[1] != vec2_np.shape[1]:
            logger.warning(f"Vector dimension mismatch: {vec1_np.shape} vs {vec2_np.shape}")
            return 0.0
        
        similarity = cosine_similarity(vec1_np, vec2_np)[0][0]
        return float(similarity)
    except Exception as e:
        logger.warning(f"Similarity computation fallback: {e}")
        return 0.0

# ============= BERT LOADING (OPTIONAL) =============
BERT_AVAILABLE = False
bert_router = None
BERTChatbotIntegration = None
BERTCookbookChatbot = None

# Check if user wants BERT
ENABLE_BERT = os.getenv('ENABLE_BERT', 'false').lower() == 'true'

if ENABLE_BERT:
    logger.info("BERT is enabled in configuration, attempting to load...")
    
    try:
        # Try to load full BERT if torch is available
        if TORCH_AVAILABLE:
            from backend.bert_chatbot.integration import BERTChatbotIntegration, router as bert_router
            from backend.bert_chatbot.model import BERTCookbookChatbot
            BERT_AVAILABLE = True
            logger.info("BERT components loaded successfully with PyTorch")
        else:
            logger.warning("PyTorch not available, trying lightweight alternative...")
            raise ImportError("PyTorch not available")
            
    except ImportError as e:
        logger.info(f"Full BERT not available: {e}")
        
        # Try sentence transformer alternative (no heavy dependencies)
        try:
            from backend.bert_chatbot.sentence_transformer_model import SentenceTransformerChatbot
            # Create a minimal integration wrapper if needed
            class MinimalBERTIntegration:
                def __init__(self, kb):
                    self.bert_chatbot = SentenceTransformerChatbot(knowledge_base=kb)
                    self.kb = kb
                    
                async def process_developer_query(self, query, context=None):
                    return await self.bert_chatbot.process_query(query, context)
            
            BERTChatbotIntegration = MinimalBERTIntegration
            BERTCookbookChatbot = SentenceTransformerChatbot
            bert_router = None  # No router for lightweight version
            BERT_AVAILABLE = True
            logger.info("Using Sentence Transformer alternative (lightweight, no heavy models)")
            
        except ImportError as e2:
            logger.warning(f"Neither BERT nor lightweight alternative available: {e2}")
            BERT_AVAILABLE = False
else:
    logger.info("BERT is disabled via configuration (ENABLE_BERT=false)")

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
        self.api_categories = {}
    
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
                "icon": "[DOCS]",
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
                "title": "TCS BaNCS API",
                "version": "1.0.0",
                "description": "TCS BaNCS Banking API Documentation"
            },
            "servers": [
                {
                    "url": "https://demoapps.tcsbancs.com",
                    "description": "BaNCS API Server"
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
        self.http_clients = {}
    
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
            app_name = "TCS BaNCS API Cookbook"
            app_version = "1.0.0"
            api_port = 7600
            host = "127.0.0.1"
            debug = False
            mcp_enabled = True
            cors_origins = ["http://localhost:7600", "http://127.0.0.1:7600"]
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
            cache_ttl = 300
            rag_enabled = False
            rag_embedding_model = "BAAI/bge-small-en-v1.5"
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
        from backend.apiweaver import BaNCSAPIWeaver
        components['apiweaver'] = True
        imports['BaNCSAPIWeaver'] = BaNCSAPIWeaver
    except ImportError:
        logger.warning("Could not import apiweaver, using fallback")
        imports['BaNCSAPIWeaver'] = SimpleAPIWeaver
    
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
BaNCSAPIWeaver = imports['BaNCSAPIWeaver']

# Log configuration
def check_and_log_configuration():
    """Check and log system configuration"""
    config = {
        "Python Version": sys.version.split()[0],
        "Platform": sys.platform,
        "BERT Requested": ENABLE_BERT,
        "BERT Available": BERT_AVAILABLE,
        "PyTorch Available": TORCH_AVAILABLE,
        "MCP Enabled": getattr(settings, 'mcp_enabled', False),
        "Debug Mode": getattr(settings, 'debug', False),
        "API Port": getattr(settings, 'api_port', 7600),
        "RAG Enabled": getattr(settings, 'rag_enabled', False)
    }
    
    logger.info("=" * 60)
    logger.info("System Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)
    
    # Warnings
    if ENABLE_BERT and not BERT_AVAILABLE:
        logger.warning("BERT was requested but could not be loaded!")
        logger.warning("The system will run without BERT features.")
    
    if sys.platform == "win32":
        logger.info("Windows detected - emojis will be replaced with text")
    
    return config

system_config = check_and_log_configuration()

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
    app.state.bert_integration = None
    app.state.rag_engine = None

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

        # ============= RAG ENGINE INITIALIZATION =============
        # Initialize RAG engine with auto-training
        if hasattr(settings, 'rag_enabled') and settings.rag_enabled:
            try:
                from backend.rag_engine import LightweightRAG

                # Embeds every endpoint once (cached under models/rag_cache; rebuilt
                # only when the endpoints, model or index version change)
                app.state.rag_engine = LightweightRAG(
                    knowledge_base=app.state.kb,
                    embedding_model=getattr(settings, 'rag_embedding_model', 'BAAI/bge-small-en-v1.5'),
                    auto_train=True
                )
                logger.info(f"RAG engine initialized with {len(app.state.rag_engine.documents)} documents")

            except Exception as e:
                logger.warning(f"Could not initialize RAG engine: {e}")
                app.state.rag_engine = None
        else:
            logger.info("RAG is disabled in configuration")
        # ============= END RAG ENGINE SECTION =============

        # Initialize response generator
        try:
            app.state.response_generator = UnifiedResponseGenerator(app.state.kb)
            logger.info("Response generator initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize response generator: {e}, using fallback")
            app.state.response_generator = SimpleResponseGenerator(app.state.kb)

        # Initialize BERT if available and enabled
        if ENABLE_BERT and BERT_AVAILABLE:
            try:
                class BERTIntegrationWrapper:
                    def __init__(self, kb):
                        # Initialize the sentence transformer with the knowledge base
                        self.bert_chatbot = BERTCookbookChatbot(knowledge_base=kb)
                        self.kb = kb
                        
                    async def process_developer_query(self, query, context=None):
                        return await self.bert_chatbot.process_query(query, context)
                
                app.state.bert_integration = BERTIntegrationWrapper(app.state.kb)
                logger.info("BERT chatbot initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize BERT chatbot: {e}")
                app.state.bert_integration = None
        else:
            if ENABLE_BERT:
                logger.info("BERT was requested but is not available")
            else:
                logger.info("BERT is disabled")
        
        # Initialize MCP if enabled
        if settings.mcp_enabled:
            try:
                app.state.apiweaver = BaNCSAPIWeaver()
                logger.info("APIWeaver (MCP) initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize APIWeaver: {e}, using fallback")
                app.state.apiweaver = SimpleAPIWeaver()
        else:
            logger.info("MCP is disabled in configuration")
            app.state.apiweaver = None
        
        if settings.mcp_enabled:
            try:
                app.state.apiweaver = BaNCSAPIWeaver()
                # Process any pending API registrations
                if hasattr(app.state.apiweaver, 'process_pending_registrations'):
                    await app.state.apiweaver.process_pending_registrations()
                logger.info("APIWeaver (MCP) initialized successfully with all APIs registered")
            except Exception as e:
                logger.warning(f"Failed to initialize APIWeaver: {e}, using fallback")
                app.state.apiweaver = SimpleAPIWeaver()

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
        logger.warning(f"Frontend directory exists but index.html not found")
else:
    logger.warning(f"Frontend directory not found at {frontend_path.absolute()}")

# ============= CORE ENDPOINTS =============
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the cookbook UI"""
    frontend_file = Path("frontend/index.html")
    if frontend_file.exists():
        return FileResponse(frontend_file)
    
    # Sanitize text for display
    status_icon = sanitize_text("✅")
    warning_icon = sanitize_text("⚠️")
    
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
            <div class="status">{status_icon} API Server is running!</div>
            {"<div class='warning'>" + warning_icon + " Running in fallback mode - some features may be limited</div>" if app.state.is_fallback_mode else ""}
            <p>Version: {settings.app_version}</p>
            <div class="links">
                <a href="/api/docs">API Documentation</a>
                <a href="/health">Health Check</a>
                <a href="/api/endpoints">API Endpoints</a>
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
            "response_generator": "fallback" if isinstance(app.state.response_generator, SimpleResponseGenerator) else "running",
            "bert": "enabled" if BERT_AVAILABLE else "disabled",
            "rag": "enabled" if app.state.rag_engine else "disabled"
        },
        "components": components_loaded,
        "stats": {
            "endpoints_loaded": len(app.state.kb.endpoints) if app.state.kb else 0,
            "mcp_tools": len(getattr(app.state.kb, 'mcp_tools', [])),
            "playground_history": len(app.state.playground_history)
        },
        "configuration": {
            "bert_requested": ENABLE_BERT,
            "bert_available": BERT_AVAILABLE,
            "pytorch_available": TORCH_AVAILABLE,
            "rag_enabled": getattr(settings, 'rag_enabled', False)
        }
    }
    
    # Add RAG statistics if available
    if hasattr(app.state, 'rag_engine') and app.state.rag_engine:
        health_status["rag_stats"] = {
            "documents": len(app.state.rag_engine.documents),
            "using_embeddings": bool(app.state.rag_engine.embedding_model),
            "model": app.state.rag_engine.embedding_model_name
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
        
        # Try RAG-based search first if available
        if hasattr(app.state, 'rag_engine') and app.state.rag_engine:
            try:
                logger.info(f"[{request_id}] Attempting RAG search")
                
                # Returns only confident matches (>= LightweightRAG.MIN_SCORE);
                # an empty result falls through to BERT/keyword search below
                rag_results = app.state.rag_engine.search(query, top_k=limit)

                if rag_results:
                    results = []
                    for doc, score in rag_results:
                        name = doc.metadata.get('endpoint_name')
                        # The index is built at startup; skip endpoints removed since then
                        ep = app.state.kb.endpoints.get(name)
                        if ep is None:
                            continue
                        results.append({
                            'endpoint_name': name,
                            'method': ep.get('method', 'GET'),
                            'path': ep.get('url', '/'),
                            'score': float(score),
                            'confidence': float(score),
                            'description': ep.get('description', ''),
                            'parameters': ep.get('parameters', {}),
                            'title': name,
                            'category': ep.get('category', 'general')
                        })

                    if results:
                        logger.info(f"[{request_id}] Found {len(results)} results via RAG")
                        return JSONResponse(content={
                            "status": "success",
                            "query": query,
                            "results": results,
                            "total": len(results),
                            "source": "rag"
                        })
            except Exception as e:
                logger.warning(f"[{request_id}] RAG search failed: {e}")
        
        # Try BERT-based search if RAG is not available and BERT is properly configured
        use_bert = (
            BERT_AVAILABLE and 
            hasattr(app.state, 'bert_integration') and 
            app.state.bert_integration and
            # os.getenv, not settings: pydantic only sees undeclared keys from the .env file, not real env vars (Docker)
            os.getenv('DISABLE_BERT_SEARCH', 'false').lower() != 'true'
        )
        
        if use_bert:
            try:
                logger.info(f"[{request_id}] Attempting BERT search")
                bert_chatbot = app.state.bert_integration.bert_chatbot
                
                # Check if BERT is properly initialized with embeddings
                if bert_chatbot and hasattr(bert_chatbot, 'api_embeddings') and bert_chatbot.api_embeddings:
                    # Generate embeddings for the query
                    query_embeddings = bert_chatbot._generate_embeddings(query)
                    
                    # Validate query embeddings
                    if query_embeddings is not None:
                        results = []
                        for endpoint_key, api_embeddings in bert_chatbot.api_embeddings.items():
                            # Skip if embeddings are invalid
                            if api_embeddings is None:
                                continue
                            
                            # Use utility function for similarity with error handling
                            score = compute_similarity(query_embeddings, api_embeddings)
                            
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
                                    'score': float(score),
                                    'confidence': float(score),
                                    'description': metadata.get('description', ''),
                                    'parameters': metadata.get('parameters', []),
                                    'title': endpoint_key,
                                    'category': metadata.get('category', 'general')
                                })
                        
                        if results:
                            # Sort by score
                            results.sort(key=lambda x: x['score'], reverse=True)
                            results = results[:limit]
                            
                            logger.info(f"[{request_id}] Found {len(results)} results via BERT")
                            return JSONResponse(content={
                                "status": "success",
                                "query": query,
                                "results": results,
                                "total": len(results),
                                "source": "bert" if TORCH_AVAILABLE else "sentence_transformer"
                            })
                else:
                    logger.warning(f"[{request_id}] BERT embeddings not initialized")
            except Exception as e:
                logger.warning(f"[{request_id}] BERT search error: {e}")
        
        # Always fall back to keyword-based search
        logger.info(f"[{request_id}] Using keyword-based search (fallback)")
        results = []
        query_lower = query.lower()
        query_words = query_lower.split()
        
        # Search in knowledge base endpoints
        if hasattr(app.state, 'kb') and app.state.kb and hasattr(app.state.kb, 'endpoints'):
            for endpoint_name, endpoint_data in app.state.kb.endpoints.items():
                # Calculate relevance score
                score = 0.0
                endpoint_lower = endpoint_name.lower()
                
                # Exact match in endpoint name
                if query_lower in endpoint_lower:
                    score += 0.8
                
                # Check description
                description = str(endpoint_data.get('description', '')).lower()
                if query_lower in description:
                    score += 0.5
                
                # Check for word matches
                for word in query_words:
                    if len(word) > 2:  # Skip very short words
                        # Partial word matching in endpoint name
                        if word in endpoint_lower:
                            score += 0.3
                        
                        # Word in description
                        if word in description:
                            score += 0.2
                        
                        # Check in URL/path
                        url = str(endpoint_data.get('url', '')).lower()
                        if word in url:
                            score += 0.2
                        
                        # Check in parameters
                        params = endpoint_data.get('parameters', {})
                        if isinstance(params, dict):
                            for param_name in params.keys():
                                if word in str(param_name).lower():
                                    score += 0.1
                        
                        # Check in category
                        category = str(endpoint_data.get('category', '')).lower()
                        if word in category:
                            score += 0.15
                
                # Normalize score
                score = min(score, 1.0)
                
                if score > 0:
                    results.append({
                        'endpoint_name': endpoint_name,
                        'method': endpoint_data.get('method', 'GET'),
                        'path': endpoint_data.get('url', '/'),
                        'score': score,
                        'confidence': score,
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
            if BERT_AVAILABLE and hasattr(app.state, 'bert_integration') and app.state.bert_integration:
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
                if BERT_AVAILABLE and hasattr(app.state, 'bert_integration') and app.state.bert_integration:
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
    class FakeRequest:
        async def json(self):
            return {"query": query, "limit": limit}
    
    return await search_knowledge(FakeRequest())

@app.post("/api/chat")
async def chat(message_data: Dict[str, Any]):
    """Enhanced chat endpoint using direct knowledge base search"""
    request_id = str(uuid.uuid4())[:8]
    
    try:
        # Parse the message
        if isinstance(message_data, dict):
            message_text = message_data.get('message', '')
            context = message_data.get('context', [])
        else:
            message_text = str(message_data)
            context = []
        
        logger.debug(f"[{request_id}] Chat request: {message_text[:100]}...")
        
        if not message_text:
            logger.warning(f"[{request_id}] Empty message received")
            return {
                "response": "Please provide a message.",
                "confidence": 0,
                "mcp_available": False,
                "api_references": []
            }
        
        message_lower = message_text.lower()
        
        # Check for greetings
        if message_lower in ['hi', 'hello', 'hey', 'greetings']:
            return {
                "response": "Hello! I'm your TCS BaNCS API assistant. I can help you with:\n\n" +
                           "• Finding API endpoints\n" +
                           "• Generating code examples\n" +
                           "• Explaining authentication\n" +
                           "• Debugging API errors\n\n" +
                           "What would you like to know about the APIs?",
                "confidence": 0.95,
                "mcp_available": settings.mcp_enabled,
                "api_references": []
            }
        
        # Check for MCP-specific queries
        if 'mcp' in message_lower or 'tool' in message_lower:
            logger.info(f"[{request_id}] MCP-related query detected")
            mcp_response = await get_mcp_tools_response(app.state)
            if mcp_response:
                mcp_response['response'] = sanitize_text(mcp_response.get('response', ''))
                logger.info(f"[{request_id}] Response source: MCP")
                return mcp_response
        
        # Check for "list all" or "show all" requests
        if any(phrase in message_lower for phrase in ['list all', 'show all', 'all api', 'all endpoint', 'show me the apis']):
            if hasattr(app.state, 'kb') and app.state.kb:
                total_endpoints = len(app.state.kb.endpoints)
                
                # Group by category
                categories = {}
                for name, data in app.state.kb.endpoints.items():
                    cat = data.get('category', 'general')
                    if cat not in categories:
                        categories[cat] = []
                    categories[cat].append({
                        'name': name,
                        'method': data.get('method', 'GET'),
                        'path': data.get('url', '/')
                    })
                
                response_text = f"I have access to **{total_endpoints} API endpoints** across {len(categories)} categories:\n\n"
                
                # Show top categories
                for cat, endpoints in list(categories.items())[:5]:
                    response_text += f"**{cat.replace('_', ' ').title()}** ({len(endpoints)} endpoints):\n"
                    for ep in endpoints[:3]:
                        response_text += f"  • `{ep['method']} {ep['path']}`\n"
                    if len(endpoints) > 3:
                        response_text += f"  _...and {len(endpoints) - 3} more_\n"
                    response_text += "\n"
                
                if len(categories) > 5:
                    response_text += f"_...and {len(categories) - 5} more categories_\n"
                
                return {
                    "response": response_text,
                    "confidence": 0.9,
                    "mcp_available": settings.mcp_enabled,
                    "api_references": [],
                    "metadata": {"source": "knowledge_base", "total_endpoints": total_endpoints}
                }
        
        # Direct knowledge base search for all other queries
        logger.info(f"[{request_id}] Using direct knowledge base search")
        
        if hasattr(app.state, 'kb') and app.state.kb:
            results = []
            query_words = message_lower.split()
            
            # Search through all endpoints
            for endpoint_name, endpoint_data in app.state.kb.endpoints.items():
                score = 0
                endpoint_lower = endpoint_name.lower()
                
                # Calculate relevance score
                for word in query_words:
                    if len(word) > 2:  # Skip very short words
                        # Check endpoint name
                        if word in endpoint_lower:
                            score += 0.4
                        
                        # Check description
                        if word in endpoint_data.get('description', '').lower():
                            score += 0.3
                        
                        # Check URL/path
                        if word in endpoint_data.get('url', '').lower():
                            score += 0.2
                        
                        # Check category
                        if word in endpoint_data.get('category', '').lower():
                            score += 0.3
                        
                        # Check parameters
                        params = endpoint_data.get('parameters', {})
                        if isinstance(params, dict):
                            for param_name in params.keys():
                                if word in param_name.lower():
                                    score += 0.1
                
                if score > 0:
                    results.append({
                        'name': endpoint_name,
                        'method': endpoint_data.get('method', 'GET'),
                        'path': endpoint_data.get('url', '/'),
                        'score': min(score, 1.0),
                        'description': endpoint_data.get('description', ''),
                        'category': endpoint_data.get('category', 'general'),
                        'parameters': endpoint_data.get('parameters', {})
                    })
            
            # Sort by relevance
            results.sort(key=lambda x: x['score'], reverse=True)
            source = "keyword"

            # Prefer confident semantic (bge) matches; the keyword scores above are the fallback.
            # ponytail: keyword loop still runs first, ~1 ms for 950 endpoints
            if getattr(app.state, 'rag_engine', None):
                try:
                    rag_hits = []
                    for doc, score in app.state.rag_engine.search(message_text, top_k=5):
                        name = doc.metadata.get('endpoint_name')
                        ep = app.state.kb.endpoints.get(name)
                        if ep is not None:
                            rag_hits.append({
                                'name': name,
                                'method': ep.get('method', 'GET'),
                                'path': ep.get('url', '/'),
                                'score': float(score),
                                'description': ep.get('description', ''),
                                'category': ep.get('category', 'general'),
                                'parameters': ep.get('parameters', {})
                            })
                    if rag_hits:
                        results, source = rag_hits, "rag"
                except Exception as e:
                    logger.warning(f"[{request_id}] RAG chat search failed, using keyword results: {e}")

            top_results = results[:5]
            
            if top_results:
                # Check if user is asking for code examples
                if any(word in message_lower for word in ['code', 'example', 'sample', 'how to', 'python', 'javascript']):
                    # Generate code example for top result
                    best_match = top_results[0]
                    code_example = generate_code_example(best_match)
                    
                    response_text = f"Here's how to use the **{best_match['name']}** endpoint:\n\n"
                    response_text += f"**Endpoint:** `{best_match['method']} {best_match['path']}`\n\n"
                    response_text += f"**Python Example:**\n```python\n{code_example}\n```\n\n"
                    
                    if len(top_results) > 1:
                        response_text += "**Related endpoints:**\n"
                        for result in top_results[1:4]:
                            response_text += f"• `{result['method']} {result['path']}`\n"
                
                else:
                    # List matching endpoints
                    response_text = f"I found **{len(top_results)} relevant API endpoints** for '{message_text}':\n\n"
                    
                    for result in top_results:
                        response_text += f"**{result['name']}**\n"
                        response_text += f"• Method: `{result['method']}`\n"
                        response_text += f"• Path: `{result['path']}`\n"
                        if result['description']:
                            response_text += f"• Description: {result['description'][:100]}...\n" if len(result['description']) > 100 else f"• Description: {result['description']}\n"
                        response_text += "\n"
                    
                    response_text += "Would you like code examples for any of these?"
                
                return {
                    "response": response_text,
                    "confidence": top_results[0]['score'],
                    "mcp_available": settings.mcp_enabled,
                    "api_references": [
                        {
                            "name": r['name'],
                            "method": r['method'],
                            "path": r['path']
                        } for r in top_results[:3]
                    ],
                    "metadata": {
                        "source": source,
                        "request_id": request_id,
                        "results_found": len(results)
                    }
                }
            else:
                # No results found
                response_text = "I couldn't find any APIs matching your query. Try asking about:\n\n"
                response_text += "• Customer management\n"
                response_text += "• Account operations\n"
                response_text += "• Transaction processing\n"
                response_text += "• Loan management\n\n"
                response_text += "Or type 'show all apis' to see all available endpoints."
                
                return {
                    "response": response_text,
                    "confidence": 0.3,
                    "mcp_available": settings.mcp_enabled,
                    "api_references": [],
                    "metadata": {
                        "source": "knowledge_base",
                        "request_id": request_id
                    }
                }
        
        # Fallback if knowledge base is not available
        return {
            "response": "Knowledge base is not available. Please try again later.",
            "confidence": 0,
            "mcp_available": settings.mcp_enabled,
            "api_references": [],
            "metadata": {"error": True, "request_id": request_id}
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] Chat endpoint error: {e}", exc_info=True)
        return {
            "response": f"Error: {str(e)}" if settings.debug else "An error occurred.",
            "confidence": 0,
            "mcp_available": False,
            "api_references": [],
            "metadata": {"error": True, "request_id": request_id}
        }

def generate_code_example(endpoint):
    """Generate a Python code example for an endpoint"""
    func_name = endpoint['name'].lower().replace(' ', '_').replace('-', '_')
    
    if endpoint['method'] == 'GET':
        code = f"""import httpx

async def {func_name}():
    headers = {{
        'entity': 'GPRDTTSTOU',
        'userId': '1',
        'languageCode': '1',
        'Content-Type': 'application/json'
    }}
    
    url = "https://demoapps.tcsbancs.com{endpoint['path']}"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        return response.json()

# Usage
import asyncio
result = asyncio.run({func_name}())
print(result)"""
    
    else:  # POST, PUT, PATCH, DELETE
        code = f"""import httpx

async def {func_name}(data):
    headers = {{
        'entity': 'GPRDTTSTOU',
        'userId': '1',
        'languageCode': '1',
        'Content-Type': 'application/json'
    }}
    
    url = "https://demoapps.tcsbancs.com{endpoint['path']}"
    
    async with httpx.AsyncClient() as client:
        response = await client.{endpoint['method'].lower()}(
            url, 
            headers=headers,
            json=data
        )
        return response.json()

# Usage
import asyncio
data = {{
    # Add your request data here
}}
result = asyncio.run({func_name}(data))
print(result)"""
    
    return code

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
                tool_descriptions.append(f"- **{tool_name}**: {tool_desc}")
        
        # Also check APIWeaver
        if hasattr(app_state, 'apiweaver') and app_state.apiweaver:
            if hasattr(app_state.apiweaver, 'apis'):
                for api_name, api_config in app_state.apiweaver.apis.items():
                    if hasattr(api_config, 'endpoints'):
                        for endpoint in api_config.endpoints:
                            tool_name = f"{api_name}_{getattr(endpoint, 'name', 'unknown')}"
                            if tool_name not in tools:
                                tools.append(tool_name)
                                tool_descriptions.append(f"- **{tool_name}**: {getattr(endpoint, 'description', 'API endpoint')}")
        
        if tools:
            response = f"**Available MCP Tools ({len(tools)} total):**\n\n"
            response += "\n".join(tool_descriptions[:10])  # Show first 10
            
            if len(tools) > 10:
                response += f"\n\n...and {len(tools) - 10} more tools."
            
            response += "\n\n**How to use MCP tools:**\n"
            response += "These tools allow AI assistants to directly interact with the BaNCS API endpoints. "
            response += "Each tool corresponds to an API endpoint and can be called with the appropriate parameters.\n\n"
            response += "**Common tools include:**\n"
            response += "- Account management (balance, details, transactions)\n"
            response += "- Customer management (create, update, search)\n"
            response += "- Loan operations (details, repayment schedules)\n"
            response += "- Transaction processing (bookings, transfers)"
            
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
                           "- API specifications haven't been loaded yet\n" +
                           "- The MCP server is still initializing\n" +
                           "- There's an issue with the API configuration\n\n" +
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
        # Sanitize emojis in recipes
        for recipe in recipes:
            if 'icon' in recipe:
                recipe['icon'] = sanitize_text(recipe['icon'])
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
            
            # Execute the tool (simplified for brevity - copy your existing logic)
            result = {"message": "MCP tool execution"}
            
        elif request.type == "api":
            # Execute direct API call
            async with httpx.AsyncClient(timeout=request.timeout) as client:
                # Build URL - only the configured BaNCS host is allowed, otherwise
                # this endpoint is an open proxy that hands our auth headers to anyone.
                if request.url.lower().startswith(("http://", "https://")):
                    url = request.url
                else:
                    url = f"{settings.api_base_url.rstrip('/')}/{request.url.lstrip('/')}"
                # Validate with httpx's own parser so the check matches what is actually sent
                target, allowed = httpx.URL(url), httpx.URL(settings.api_base_url)
                if (target.scheme, target.host, target.port) != (allowed.scheme, allowed.host, allowed.port):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Only requests to {settings.api_base_url} are allowed"
                    )

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
            "request": request.dict(),
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
        raise
        
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

# ============= HOPPSCOTCH INTEGRATION ENDPOINTS =============
@app.get("/api/hoppscotch/collections")
async def export_to_hoppscotch():
    """Export API endpoints as Hoppscotch collections"""
    try:
        collections = []
        
        # Group endpoints by category
        categories = {}
        
        if hasattr(app.state, 'kb') and app.state.kb:
            for endpoint_name, endpoint_data in app.state.kb.endpoints.items():
                category = endpoint_data.get('category', 'General')
                if category not in categories:
                    categories[category] = []
                
                # Build request object for Hoppscotch
                request = {
                    "name": endpoint_name,
                    "method": endpoint_data.get('method', 'GET'),
                    "endpoint": endpoint_data.get('url', '/'),
                    "headers": [
                        {"key": "entity", "value": settings.bancs_entity},
                        {"key": "userId", "value": str(settings.bancs_user_id)},
                        {"key": "languageCode", "value": str(settings.bancs_language_code)},
                        {"key": "Content-Type", "value": "application/json"}
                    ],
                    "params": [],
                    "body": {
                        "contentType": "application/json",
                        "body": "{}"
                    }
                }
                
                # Add parameters
                params = endpoint_data.get('parameters', {})
                if isinstance(params, dict):
                    for param_name, param_data in params.items():
                        if isinstance(param_data, dict):
                            required = param_data.get('required', False)
                            request["params"].append({
                                "key": param_name,
                                "value": "",
                                "active": required
                            })
                
                categories[category].append(request)
        
        # Create collection for each category
        for category, requests in categories.items():
            collection = HoppscotchCollection(
                name=f"TCS BaNCS - {category}",
                requests=requests
            )
            collections.append(collection.dict())
        
        return {
            "collections": collections,
            "total": sum(len(cat) for cat in categories.values()),
            "categories": list(categories.keys())
        }
        
    except Exception as e:
        logger.error(f"Hoppscotch export error: {e}")
        raise HTTPException(status_code=500, detail="Failed to export collections")

@app.post("/api/hoppscotch/import")
async def import_from_hoppscotch(file: UploadFile = File(...)):
    """Import Hoppscotch collection and add to knowledge base"""
    try:
        content = await file.read()
        data = json.loads(content)
        
        imported_count = 0
        
        # Process collection
        if 'requests' in data:
            for request in data['requests']:
                endpoint_name = request.get('name', f"Endpoint_{imported_count}")
                endpoint_data = {
                    'method': request.get('method', 'GET'),
                    'url': request.get('endpoint', '/'),
                    'description': request.get('description', ''),
                    'parameters': {},
                    'category': 'imported'
                }
                
                # Extract parameters from request
                if 'params' in request:
                    for param in request['params']:
                        param_name = param.get('key')
                        if param_name:
                            endpoint_data['parameters'][param_name] = {
                                'type': 'string',
                                'required': param.get('active', False),
                                'description': param.get('description', '')
                            }
                
                # Add to knowledge base
                if hasattr(app.state, 'kb') and app.state.kb:
                    app.state.kb.endpoints[endpoint_name] = endpoint_data
                    imported_count += 1
        
        return {
            "success": True,
            "imported": imported_count,
            "message": f"Imported {imported_count} endpoints from Hoppscotch collection"
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    except Exception as e:
        logger.error(f"Hoppscotch import error: {e}")
        raise HTTPException(status_code=500, detail="Failed to import collection")

# ============= WEBSOCKET ENDPOINTS =============
class ConnectionManager:
    """WebSocket connection manager"""
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_info: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_info[websocket] = {
            "connected_at": datetime.now(timezone.utc).isoformat(),
            "id": str(uuid.uuid4())[:8]
        }
        logger.info(f"WebSocket connected: {self.connection_info[websocket]['id']}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.connection_info:
            logger.info(f"WebSocket disconnected: {self.connection_info[websocket]['id']}")
            del self.connection_info[websocket]
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except:
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await manager.connect(websocket)
    ws_id = manager.connection_info[websocket]['id']
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "system",
            "message": "Connected to TCS BaNCS API Cookbook",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bert_available": BERT_AVAILABLE,
            "mcp_available": settings.mcp_enabled
        })
        
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            logger.info(f"[WS-{ws_id}] Received: {message_data.get('message', '')[:100]}")
            
            # Process message through chat system
            response = await chat(message_data)
            
            # Send response back
            await websocket.send_json({
                "type": "response",
                "data": response,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"[WS-{ws_id}] Client disconnected")
    except Exception as e:
        logger.error(f"[WS-{ws_id}] WebSocket error: {e}")
        manager.disconnect(websocket)

@app.websocket("/ws/playground")
async def websocket_playground(websocket: WebSocket):
    """WebSocket endpoint for playground real-time updates"""
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    
    logger.info(f"Playground WebSocket connected: {session_id}")
    
    try:
        # Store session
        app.state.playground_sessions[session_id] = {
            "websocket": websocket,
            "connected_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Send initial state
        await websocket.send_json({
            "type": "init",
            "session_id": session_id,
            "history": app.state.playground_history[:10]
        })
        
        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            # Process playground request
            if request_data.get('type') == 'execute':
                try:
                    # Parse and execute request
                    request = PlaygroundRequest(**request_data.get('request', {}))
                    result = await execute_playground_request(request)
                    
                    # Send result
                    await websocket.send_json({
                        "type": "result",
                        "data": result
                    })
                    
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "error": str(e)
                    })
            
    except WebSocketDisconnect:
        logger.info(f"Playground WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Playground WebSocket error: {e}")
    finally:
        # Clean up session
        if session_id in app.state.playground_sessions:
            del app.state.playground_sessions[session_id]

# ============= DEVELOPER TOOLS ENDPOINTS =============
@app.get("/api/developer/bert-status")
async def get_bert_status():
    """Get detailed BERT status information"""
    status = {
        "enabled": ENABLE_BERT,
        "available": BERT_AVAILABLE,
        "pytorch_available": TORCH_AVAILABLE,
        "model_type": None,
        "embeddings_count": 0,
        "memory_usage": None
    }
    
    if BERT_AVAILABLE and hasattr(app.state, 'bert_integration') and app.state.bert_integration:
        try:
            bert_chatbot = app.state.bert_integration.bert_chatbot
            
            if TORCH_AVAILABLE:
                status["model_type"] = "full_bert"
                # Get model info if available
                if hasattr(bert_chatbot, 'model'):
                    status["model_info"] = {
                        "name": bert_chatbot.model.__class__.__name__,
                        "device": str(bert_chatbot.device) if hasattr(bert_chatbot, 'device') else "cpu"
                    }
            else:
                status["model_type"] = "lightweight_sentence_transformer"
            
            # Get embeddings count
            if hasattr(bert_chatbot, 'api_embeddings'):
                status["embeddings_count"] = len(bert_chatbot.api_embeddings)
            
            # Memory usage (if PyTorch available)
            if TORCH_AVAILABLE and torch is not None:
                if torch.cuda.is_available():
                    status["memory_usage"] = {
                        "cuda_allocated": f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB",
                        "cuda_reserved": f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB"
                    }
            
        except Exception as e:
            status["error"] = str(e)
    
    return status

@app.post("/api/developer/reload-knowledge")
async def reload_knowledge_base():
    """Reload the knowledge base"""
    try:
        logger.info("Reloading knowledge base...")
        
        # Reload specifications
        if hasattr(app.state, 'kb'):
            await app.state.kb.load_specifications()
            
            # Reinitialize BERT embeddings if available
            if BERT_AVAILABLE and hasattr(app.state, 'bert_integration'):
                bert_chatbot = app.state.bert_integration.bert_chatbot
                if hasattr(bert_chatbot, 'initialize_api_embeddings'):
                    bert_chatbot.initialize_api_embeddings()
            
            logger.info("Knowledge base reloaded successfully")
            
            return {
                "success": True,
                "message": "Knowledge base reloaded",
                "endpoints_count": len(app.state.kb.endpoints),
                "bert_updated": BERT_AVAILABLE
            }
        else:
            return {
                "success": False,
                "message": "Knowledge base not available"
            }
            
    except Exception as e:
        logger.error(f"Failed to reload knowledge base: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/developer/logs")
async def get_recent_logs(level: str = "INFO", limit: int = 100):
    """Get recent application logs (debug mode only - logs contain request data)"""
    if not getattr(settings, 'debug', False):
        raise HTTPException(status_code=403, detail="Log access is only available in debug mode")
    try:
        log_file = Path("logs/app.log")
        
        if not log_file.exists():
            return {
                "logs": [],
                "message": "Log file not found"
            }
        
        # Read last N lines from log file
        lines = []
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            # Read file in reverse
            lines = f.readlines()[-limit:]
        
        # Filter by level if specified
        level_priority = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3,
            "CRITICAL": 4
        }
        
        min_priority = level_priority.get(level.upper(), 1)
        
        filtered_logs = []
        for line in lines:
            for log_level, priority in level_priority.items():
                if log_level in line and priority >= min_priority:
                    filtered_logs.append(line.strip())
                    break
        
        return {
            "logs": filtered_logs,
            "total": len(filtered_logs),
            "level": level
        }
        
    except Exception as e:
        logger.error(f"Failed to read logs: {e}")
        return {
            "logs": [],
            "error": str(e)
        }

@app.post("/api/developer/test-endpoint")
async def test_endpoint(endpoint_name: str, params: Optional[Dict[str, Any]] = None):
    """Test a specific endpoint with optional parameters"""
    try:
        if not hasattr(app.state, 'kb') or endpoint_name not in app.state.kb.endpoints:
            raise HTTPException(status_code=404, detail=f"Endpoint '{endpoint_name}' not found")
        
        endpoint_data = app.state.kb.endpoints[endpoint_name]
        
        # Build test request
        test_request = PlaygroundRequest(
            type="api",
            method=endpoint_data.get('method', 'GET'),
            url=endpoint_data.get('url', '/'),
            params=params,
            use_auth=True
        )
        
        # Execute test
        result = await execute_playground_request(test_request)
        
        return {
            "endpoint": endpoint_name,
            "test_result": result,
            "endpoint_data": endpoint_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Test endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")

# ============= ERROR HANDLERS =============
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An unexpected error occurred",
            "path": str(request.url)
        }
    )

# ============= MAIN ENTRY POINT =============
if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    host = getattr(settings, 'host', '0.0.0.0')
    port = getattr(settings, 'api_port', 7600)
    debug = getattr(settings, 'debug', False)
    
    logger.info("=" * 60)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Host: {host}:{port}")
    logger.info(f"Debug Mode: {debug}")
    logger.info(f"BERT: {'Enabled' if BERT_AVAILABLE else 'Disabled'}")
    logger.info(f"MCP: {'Enabled' if settings.mcp_enabled else 'Disabled'}")
    logger.info("=" * 60)
    
    # Run server
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug",
        access_log=True
    )