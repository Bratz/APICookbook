"""
TCS BaNCS API Cookbook - Complete Main Application with Hoppscotch Integration
Enhanced with improved error handling, validation, and fallback mechanisms
BERT/PyTorch are now completely optional
"""

from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import asyncio
import contextlib
import logging
import json
import hmac
import os
import re
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

from backend import api_lookup, build_proxy, build_routes
from backend.agent_guard import AgentGuard

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

# The MCP endpoint (decided 2026-11-09): APIWeaver's four tools over streamable HTTP at /api/mcp/, for any
# MCP client, behind a per-IP limit. Writes are held until a person approves them on the site (apiweaver.py).
MCP_WEAVER = None
MCP_HTTP = None
if settings.mcp_enabled:
    try:
        MCP_WEAVER = BaNCSAPIWeaver()
        MCP_HTTP = MCP_WEAVER.mcp.http_app(path="/") if getattr(MCP_WEAVER, "mcp", None) else None
    except Exception as e:  # the site works without MCP; the badge says so
        logger.warning(f"MCP unavailable: {e}")
        MCP_WEAVER = SimpleAPIWeaver()

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
        
        # MCP: the weaver was built at import (it is mounted at /api/mcp); register the APIs it loaded.
        app.state.apiweaver = MCP_WEAVER
        if hasattr(MCP_WEAVER, 'process_pending_registrations'):
            await MCP_WEAVER.process_pending_registrations()

        # Initialize playground state
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
    
    # Build & run's BaNCS proxy (docs/decode-fork-plan.md, F2a): a second server in this process, so it
    # shares the knowledge base and the approvals the /api/build routes record. Off unless a port is set.
    app.state.build_proxy = None
    proxy_port = int(os.getenv("BUILD_PROXY_PORT", "0") or 0)
    if proxy_port:
        app.state.build_proxy = build_proxy.server(proxy_port)
        app.state.build_proxy_task = asyncio.create_task(app.state.build_proxy.serve())
        logger.info(f"Build proxy listening on :{proxy_port}")

    if MCP_HTTP is not None:
        async with MCP_HTTP.lifespan(MCP_HTTP):  # FastMCP's session manager lives as long as the app
            yield  # App runs here
    else:
        yield  # App runs here

    if app.state.build_proxy:
        app.state.build_proxy.should_exit = True
        with contextlib.suppress(Exception):
            await asyncio.wait_for(app.state.build_proxy_task, timeout=5)
    
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
app.include_router(build_routes.router)  # Build & run, operator-only until F3/F4


# ============= MIDDLEWARE =============
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

_MCP_GUARD = None


@app.middleware("http")
async def mcp_rate_limit(request: Request, call_next):
    """MCP is open to anyone: cap requests per IP (each tool call is one POST to /api/mcp/)."""
    global _MCP_GUARD
    if request.url.path == "/api/mcp":
        # Vercel's /api/:path* rewrite cannot carry a trailing slash: serve /api/mcp itself, no redirect.
        request.scope["path"], request.scope["raw_path"] = "/api/mcp/", b"/api/mcp/"
    if request.method == "POST" and request.url.path.rstrip("/") == "/api/mcp":
        if _MCP_GUARD is None:
            _MCP_GUARD = AgentGuard(os.getenv("MCP_DATA_DIR", "data/mcp"),
                                    per_hour=int(os.getenv("MCP_RATE_PER_HOUR", "300")))
        if _MCP_GUARD.check(build_routes._client_ip(request)):
            return JSONResponse(status_code=429, content={"error": "rate_limited",
                                "message": "Too many MCP requests from this address; try again later."})
    return await call_next(request)


def operator_only(authorization: str = Header("")):
    """Routes that change the shared knowledge base, or call the bank on the server's behalf: RUNNER_TOKEN only.
    (Caddy also blocks the two knowledge-base writes; this covers every route, behind any proxy.)"""
    token = os.getenv("RUNNER_TOKEN", "")
    if len(token) < 32 or not hmac.compare_digest(authorization.encode(), f"Bearer {token}".encode()):
        raise HTTPException(status_code=403, detail="Operator only.")


class WriteDecision(BaseModel):
    approve: bool


@app.get("/api/mcp-writes/{write_id}")
async def mcp_held_write(write_id: str):
    """The approval page: one write an MCP client asked for, waiting for a person."""
    write = MCP_WEAVER.held_write(write_id) if hasattr(MCP_WEAVER, "held_write") else None
    if write is None:
        raise HTTPException(status_code=404, detail="No such write (wrong link, or it was cleared).")
    return write


@app.post("/api/mcp-writes/{write_id}")
async def mcp_decide_write(write_id: str, body: WriteDecision):
    """Approve (sends it to the bank now) or reject. Holding the link is the authority to decide."""
    write = await MCP_WEAVER.decide_write(write_id, body.approve) if hasattr(MCP_WEAVER, "decide_write") else None
    if write is None:
        raise HTTPException(status_code=409, detail="This write is not waiting any more (decided or expired).")
    return write


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
            "mcp_tools": len(getattr(app.state.kb, 'mcp_tools', []))
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

@app.post("/api/knowledge/add", dependencies=[Depends(operator_only)])
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

@app.delete("/api/knowledge/{endpoint_name}", dependencies=[Depends(operator_only)])
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

def _example_body(schema: dict, node=None, depth: int = 0):
    """A skeleton of a request body from its schema: field names with empty values of the right type."""
    node = schema if node is None else node
    for _ in range(20):  # follow $refs into the inlined definitions
        ref = node.get("$ref", "") if isinstance(node, dict) else ""
        if not ref.startswith("#/definitions/"):
            break
        node = schema.get("definitions", {}).get(ref.split("/")[-1], {})
    kind = node.get("type")
    if kind == "object" or "properties" in node:
        if depth >= 3:
            return {}
        # ponytail: first 25 fields per object; the biggest BaNCS bodies have hundreds
        return {k: _example_body(schema, v, depth + 1) for k, v in list(node.get("properties", {}).items())[:25]}
    if kind == "array":
        return [_example_body(schema, node.get("items", {}), depth + 1)]
    return {"integer": 0, "number": 0.0, "boolean": False}.get(kind, "")


def generate_code_example(endpoint):
    """Generate a Python code example for an endpoint (the search chat's template; the agent writes real code)."""
    import pprint
    func_name = endpoint['name'].lower().replace(' ', '_').replace('-', '_')
    path = endpoint['path']
    if not path.startswith('/Core/'):  # Swagger paths are relative to the spec's basePath /Core
        path = '/Core' + path
    url = settings.api_base_url.rstrip('/') + path
    headers = (f"""    headers = {{
        'entity': '{settings.bancs_entity}',
        'userId': '{settings.bancs_user_id}',
        'languageCode': '{settings.bancs_language_code}',
        'Accept': 'application/json',
        'Accesstoken': os.environ.get('BANCS_ACCESS_TOKEN', ''),  # your BaNCS access token
        'Content-Type': 'application/json'
    }}""")
    placeholder = "\n# Replace the {placeholders} in the URL with real values." if '{' in path else ""

    if endpoint['method'] == 'GET':
        code = f"""import os
import httpx

async def {func_name}():
{headers}

    url = "{url}"

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        return response.json()

# Usage{placeholder}
import asyncio
result = asyncio.run({func_name}())
print(result)"""

    else:  # POST, PUT, PATCH, DELETE
        kb = getattr(app.state, 'kb', None)
        schema = kb.resolved_body_schema(endpoint['name']) if kb and hasattr(kb, 'resolved_body_schema') else None
        if schema:
            body = "# Fields from the API spec; fill in real values.\ndata = " + pprint.pformat(
                _example_body(schema), indent=4, width=100, sort_dicts=False)
        else:
            body = "data = {\n    # Add your request data here\n}"
        code = f"""import os
import httpx

async def {func_name}(data):
{headers}

    url = "{url}"

    async with httpx.AsyncClient() as client:
        response = await client.{endpoint['method'].lower()}(
            url,
            headers=headers,
            json=data
        )
        return response.json()

# Usage{placeholder}
import asyncio
{body}
result = asyncio.run({func_name}(data))
print(result)"""
    
    return code

async def _mcp_tools() -> list[dict]:
    """The tools the MCP server really serves (/api/mcp and the Playground call the same server).
    Every BaNCS endpoint goes through call_api; there is no tool per endpoint."""
    weaver = getattr(app.state, "apiweaver", None)
    if not settings.mcp_enabled or not getattr(weaver, "mcp", None):
        return []
    from fastmcp import Client
    async with Client(weaver.mcp) as mcp_client:
        listed = await mcp_client.list_tools()
    return [{"name": t.name, "description": (t.description or "").strip().split("\n")[0],
             "input_schema": getattr(t, "input_schema", None) or t.inputSchema} for t in listed]


async def get_mcp_tools_response(app_state) -> Optional[Dict[str, Any]]:
    """Chat answer to a question about MCP: the real tools and how to call an endpoint."""
    if not settings.mcp_enabled:
        return {"response": "The MCP server is turned off on this deployment.",
                "confidence": 1.0, "mcp_available": False, "api_references": []}
    try:
        tools = await _mcp_tools()
    except Exception as e:
        logger.error(f"MCP tools error: {e}")
        tools = []
    if not tools:
        return {"response": "The MCP server has not loaded its tools. Try again in a minute.",
                "confidence": 0.5, "mcp_available": False, "api_references": []}
    lines = "\n".join(f"- **{t['name']}**: {t['description']}" for t in tools)
    response = (f"**The MCP server has {len(tools)} tools:**\n\n{lines}\n\n"
                "Each BaNCS endpoint is called through `call_api`, for example "
                '`{"api_name": "TCS_BaNCS", "endpoint_name": "get_account_balance", '
                '"parameters": {"accountReference": "101000000101814"}}`. '
                "Use `search_endpoints` or `get_api_schema` to find an endpoint's name and parameters. "
                "Connect an MCP client to https://apicookbook.vercel.app/api/mcp.")
    return {"response": response, "confidence": 1.0, "mcp_available": True, "api_references": [],
            "metadata": {"total_tools": len(tools), "tools_list": [t["name"] for t in tools]}}

@app.get("/api/endpoints")
async def list_endpoints(category: Optional[str] = None):
    """List all API endpoints with enhanced details"""
    try:
        endpoints = await app.state.kb.list_endpoints(category)
        return endpoints or []
    except Exception as e:
        logger.error(f"Endpoints error: {e}")
        return []

@app.get("/api/body-schema")
async def get_body_schema(method: str, url: str):
    """JSON Schema for the request body of the endpoint at this method + URL (the Playground editor uses it).

    Matches the URL's path against the Swagger path templates ({param} matches one segment) and inlines
    every definition the body $ref reaches, so the result is a self-contained schema.
    """
    kb = app.state.kb
    path = httpx.URL(url).path if "://" in url else url.split("?")[0]
    if path.startswith("/Core/"):  # the spec's basePath; Swagger paths omit it
        path = path[len("/Core"):]
    path = "/" + path.strip("/")

    for name, ep in kb.endpoints.items():
        body = ep.get("body_schema")
        if not body or ep.get("method", "").upper() != method.upper():
            continue
        template = re.escape(ep.get("url", "")).replace(r"\{", "{").replace(r"\}", "}")
        pattern = re.sub(r"\{[^/]+\}", r"[^/]+", template)
        if not re.fullmatch(pattern, path.rstrip("/") or "/"):
            continue
        return {"endpoint": name, "schema": kb.resolved_body_schema(name)}

    raise HTTPException(status_code=404, detail="No request-body schema for this method and URL")


@app.get("/api/endpoint/{name}")
async def endpoint_card(name: str):
    """One endpoint, straight from the spec (no model): what the sidebar and recipe cards show.

    Method, full path, description, parameters, body schema with an example body, and a curl command
    with the demo host's headers. Swagger operations use the code agent's own lookups; the Postman
    collection's nicknames have a ready URL and their parameters as given.
    """
    kb = app.state.kb
    ep = kb.endpoints.get(name)
    if not ep:
        raise HTTPException(status_code=404, detail=f"No endpoint named {name}")
    path = ep.get("url", "")
    if not path.startswith("/Core/"):
        path = "/Core" + path  # Swagger paths are relative to the spec's basePath
    schema, params = None, {}
    if ep.get("source") == "swagger":
        params = api_lookup.get_endpoint(kb, name).get("parameters", {})
        schema = kb.resolved_body_schema(name)
    else:
        raw = ep.get("parameters")
        if isinstance(raw, dict):
            params = {k: {f: v.get(f) for f in ("type", "required", "description") if isinstance(v, dict) and v.get(f) is not None}
                      for k, v in raw.items()}
    example = _example_body(schema) if schema else None
    # Required query parameters go in the URL as {placeholders}, so the copied curl is complete.
    required_query = [k for k, v in params.items() if v.get("in") == "query" and v.get("required")]
    url = path + ("?" + "&".join(f"{k}={{{k}}}" for k in required_query) if required_query else "")
    base = settings.api_base_url.rstrip("/")
    curl = (f"curl -X {ep['method']} '{base}{url}' \\\n  -H 'entity: {settings.bancs_entity}' \\\n"
            f"  -H 'userId: {settings.bancs_user_id}' \\\n  -H 'languageCode: {settings.bancs_language_code}' \\\n"
            "  -H 'Accept: application/json' \\\n  -H \"Accesstoken: $BANCS_ACCESS_TOKEN\"")
    if example is not None:
        curl += f" \\\n  -H 'Content-Type: application/json' \\\n  -d '{json.dumps(example)}'"
    return {"name": name, "method": ep["method"], "path": path, "url": url, "description": ep.get("description") or "",
            "area": ep.get("category") or "", "source": ep.get("source") or "collection", "parameters": params,
            "body_schema": schema, "example_body": example, "curl": curl}


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
    """The tools the MCP server serves, with their input schemas."""
    if not settings.mcp_enabled:
        return {"error": "MCP is disabled", "tools": [], "total": 0, "enabled": False}
    try:
        tools = await _mcp_tools()
    except Exception as e:
        logger.error(f"MCP tools error: {e}")
        return {"tools": [], "total": 0, "error": str(e), "enabled": True}
    return {"tools": tools, "total": len(tools), "enabled": True,
            "fallback": isinstance(app.state.apiweaver, SimpleAPIWeaver)}

# ============= ENHANCED PLAYGROUND ENDPOINTS =============
@app.post("/api/playground/execute")
async def execute_playground_request(request: PlaygroundRequest, token: str = Depends(build_proxy.visitor_token)):
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
            
            from fastmcp import Client
            from backend.apiweaver import BANK_TOKEN
            BANK_TOKEN.set(token)  # call_api reads it: in-process calls have no HTTP headers of their own
            async with Client(app.state.apiweaver.mcp) as mcp_client:  # in-process: the same tools /api/mcp serves
                tool_result = await mcp_client.call_tool(request.tool_name or "", request.tool_params or {},
                                                         raise_on_error=False)
            result = tool_result.structured_content or {"content": [getattr(c, "text", str(c)) for c in tool_result.content]}
            
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
                # The Reference and the guides show paths without the spec's basePath; the bank 404s without it.
                if not target.path.startswith("/Core/"):
                    target = target.copy_with(path="/Core" + target.path)
                    url = str(target)
                placeholder = re.search(r"\{[^}/]+\}", httpx.URL(url).path.replace("%7B", "{").replace("%7D", "}"))
                if placeholder:
                    raise HTTPException(status_code=400,
                                        detail=f"Fill in {placeholder.group(0)} in the URL with a real value first.")

                logger.info(f"[{request_id}] Executing API call: {request.method} {url}")
                
                # Add auth headers if needed
                headers = request.headers or {}
                if request.use_auth:
                    headers.update({
                        "entity": settings.bancs_entity,
                        "userId": str(settings.bancs_user_id),
                        "languageCode": str(settings.bancs_language_code)
                    })
                # The bank answers XML unless asked for JSON; an Accept the visitor typed wins.
                if not any(k.lower() == "accept" for k in headers):
                    headers["Accept"] = "application/json"
                if not any(k.lower() == "accesstoken" for k in headers):  # an Accesstoken row typed in wins
                    build_proxy.require_token(token)
                    if token:
                        headers["Accesstoken"] = token
                
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
        
        
        return {
            "success": False,
            "error": str(e),
            "duration": int((time.time() - start_time) * 1000),
            "request_id": request_id
        }

# ============= HOPPSCOTCH INTEGRATION ENDPOINTS =============
@app.post("/api/hoppscotch/import", dependencies=[Depends(operator_only)])
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

# ============= DEVELOPER TOOLS ENDPOINTS =============
@app.post("/api/developer/reload-knowledge", dependencies=[Depends(operator_only)])
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

@app.post("/api/developer/test-endpoint", dependencies=[Depends(operator_only)])
async def test_endpoint(endpoint_name: str, params: Optional[Dict[str, Any]] = None,
                        token: str = Depends(build_proxy.visitor_token)):
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
        result = await execute_playground_request(test_request, token)
        
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
if MCP_HTTP is not None:
    app.mount("/api/mcp", MCP_HTTP)  # MCP clients: https://apicookbook.vercel.app/api/mcp


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