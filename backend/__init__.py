"""
TCS BaNCS API Cookbook Backend Package
Unified chatbot and MCP server implementation
"""

from typing import TYPE_CHECKING

__version__ = "2.0.0"
__author__ = "TCS BaNCS Development Team"

# Version exports
VERSION = __version__
API_VERSION = "v2"

# Module exports
from .config import settings, Settings
from .knowledge_base import UnifiedAPIKnowledgeBase
from .response_generator import UnifiedResponseGenerator
from .apiweaver import BaNCSAPIWeaver
from .spec_loader import APISpecificationLoader

# Conditional imports for type checking
if TYPE_CHECKING:
    from .main import app
    from .mcp_server import IntegratedMCPServer
    from .models import (
        ChatMessage,
        ChatResponse,
        MCPToolRequest,
        MCPToolResponse,
        CookbookRecipe,
        APIEndpoint
    )

# Package metadata
__all__ = [
    # Core components
    "settings",
    "Settings",
    "UnifiedAPIKnowledgeBase",
    "UnifiedResponseGenerator",
    "BaNCSAPIWeaver",
    "APISpecificationLoader",
    
    # Version info
    "VERSION",
    "API_VERSION",
    
    # Application (when imported)
    "app",
]

# Initialize logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
