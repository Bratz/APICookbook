"""
MCP (Model Context Protocol) Integration Package
Provides MCP server and client implementations for TCS BaNCS APIs
"""

from typing import List, Dict, Any, Optional

__version__ = "2.0.0"

# Check if MCP is available
try:
    import mcp
    from mcp import Server, Tool, Resource
    from mcp.types import TextContent, CallToolResult
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Server = None
    Tool = None
    Resource = None
    TextContent = None
    CallToolResult = None

# Module exports
from .client import MCPClient
from .tool_generator import ToolGenerator

# Package exports
__all__ = [
    # Availability flag
    "MCP_AVAILABLE",
    
    # MCP types (if available)
    "Server",
    "Tool",
    "Resource",
    "TextContent",
    "CallToolResult",
    
    # Custom implementations
    "MCPClient",
    "ToolGenerator",
    
    # Utility functions
    "create_mcp_tool",
    "register_api_as_tool",
    "get_available_tools",
]

# Utility functions
def create_mcp_tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    handler: Optional[Any] = None
) -> Optional[Tool]:
    """
    Create an MCP tool from API specification
    
    Args:
        name: Tool name
        description: Tool description
        parameters: Parameter schema
        handler: Optional tool handler function
        
    Returns:
        MCP Tool instance or None if MCP not available
    """
    if not MCP_AVAILABLE or not Tool:
        return None
    
    return Tool(
        name=name,
        description=description,
        inputSchema={
            "type": "object",
            "properties": parameters,
            "required": [
                k for k, v in parameters.items() 
                if v.get("required", False)
            ]
        }
    )

def register_api_as_tool(
    api_endpoint: Dict[str, Any],
    tool_prefix: str = "bancs"
) -> Optional[Tool]:
    """
    Register an API endpoint as an MCP tool
    
    Args:
        api_endpoint: API endpoint specification
        tool_prefix: Prefix for tool name
        
    Returns:
        MCP Tool instance or None
    """
    if not MCP_AVAILABLE:
        return None
    
    tool_name = f"{tool_prefix}_{api_endpoint['name'].lower().replace(' ', '_')}"
    
    return create_mcp_tool(
        name=tool_name,
        description=api_endpoint.get('description', ''),
        parameters=api_endpoint.get('parameters', {})
    )

def get_available_tools() -> List[str]:
    """
    Get list of available MCP tools
    
    Returns:
        List of tool names
    """
    if not MCP_AVAILABLE:
        return []
    
    # This would be populated from the actual tool registry
    return [
        "bancs_get_account_balance",
        "bancs_get_account_details",
        "bancs_create_customer",
        "bancs_create_account",
        "bancs_get_loan_details",
        "bancs_create_transaction"
    ]

# Initialize logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
