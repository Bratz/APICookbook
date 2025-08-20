# backend/apiweaver.py
"""
Advanced APIWeaver implementation with full MCP support
Complete replacement with all features from the advanced version
"""

import json
import asyncio
import inspect
from typing import Dict, Any, Optional, List, Union
from urllib.parse import urljoin, quote
import httpx
import os
# Set required environment variable for FastMCP 2.8.1+
os.environ.setdefault('FASTMCP_LOG_LEVEL', 'INFO')
from fastmcp import FastMCP, Context
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)

# ===================== Models =====================

class AuthType(str, Enum):
    """Authentication types."""
    NONE = "none"
    BEARER = "bearer"
    API_KEY = "api_key"
    BASIC = "basic"
    CUSTOM = "custom"

class RequestParam(BaseModel):
    """API request parameter definition."""
    name: str
    type: str = "string"
    location: str = "query"  # query, path, header, body
    required: bool = False
    description: Optional[str] = None
    default: Optional[Any] = None
    enum: Optional[List[str]] = None

class APIEndpoint(BaseModel):
    """API endpoint definition."""
    name: str
    method: str = "GET"
    path: str
    description: Optional[str] = None
    params: List[RequestParam] = Field(default_factory=list)
    headers: Optional[Dict[str, str]] = None
    timeout: float = 30.0

class AuthConfig(BaseModel):
    """Authentication configuration."""
    type: AuthType = AuthType.NONE
    bearer_token: Optional[str] = None
    api_key: Optional[str] = None
    api_key_header: Optional[str] = None
    api_key_param: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    custom_headers: Optional[Dict[str, str]] = None

class APIConfig(BaseModel):
    """Complete API configuration."""
    name: str
    base_url: str
    description: Optional[str] = None
    version: str = "1.0"
    auth: Optional[AuthConfig] = None
    headers: Optional[Dict[str, str]] = None
    endpoints: List[APIEndpoint] = Field(default_factory=list)

# ===================== Main APIWeaver Class =====================

class BAnCSAPIWeaver:
    """Main server that creates MCP tools from API configurations."""
    
    def __init__(self, name: str = "TCS-BAnCS-APIWeaver"):
        self.mcp = FastMCP(name)
        self.apis: Dict[str, APIConfig] = {}
        self.http_clients: Dict[str, httpx.AsyncClient] = {}
        self._setup_core_tools()
        self._initialize_default_apis()
    
    def _initialize_default_apis(self):
        """Initialize with default BAnCS APIs and load from files."""
        # Default BAnCS configuration
        bancs_config = APIConfig(
            name="TCS_BAnCS",
            base_url="https://demoapps.tcsbancs.com",
            description="TCS BAnCS Banking APIs",
            auth=AuthConfig(
                type=AuthType.CUSTOM,
                custom_headers={
                    "entity": "GPRDTTSTOU",
                    "userId": "1",
                    "languageCode": "1"
                }
            ),
            endpoints=[
                APIEndpoint(
                    name="get_account_balance",
                    method="GET",
                    path="/Core/accountManagement/account/balanceDetails/{accountReference}",
                    description="Get account balance details",
                    params=[
                        RequestParam(
                            name="accountReference",
                            type="string",
                            location="path",
                            required=True,
                            description="15-digit account reference number"
                        )
                    ]
                ),
                APIEndpoint(
                    name="get_customer_details",
                    method="GET",
                    path="/Core/customerManagement/customer/viewDetails",
                    description="Get customer details",
                    params=[
                        RequestParam(
                            name="CustomerID",
                            type="string",
                            location="query",
                            required=True,
                            description="Customer ID"
                        )
                    ]
                ),
                APIEndpoint(
                    name="create_customer",
                    method="POST",
                    path="/Core/customerManagement/customer",
                    description="Create a new customer",
                    params=[
                        RequestParam(
                            name="customerType",
                            type="integer",
                            location="body",
                            required=True,
                            description="Customer type (1=Individual, 2=Corporate)"
                        ),
                        RequestParam(
                            name="customerFullName",
                            type="string",
                            location="body",
                            required=True,
                            description="Full name of customer"
                        ),
                        RequestParam(
                            name="nationality",
                            type="string",
                            location="body",
                            required=True,
                            description="Nationality code"
                        )
                    ]
                ),
                APIEndpoint(
                    name="create_transaction",
                    method="POST",
                    path="/Core/bookingManagement/booking",
                    description="Create a transaction/booking",
                    params=[
                        RequestParam(
                            name="from_account",
                            type="string",
                            location="body",
                            required=True,
                            description="Source account"
                        ),
                        RequestParam(
                            name="to_account",
                            type="string",
                            location="body",
                            required=True,
                            description="Destination account"
                        ),
                        RequestParam(
                            name="amount",
                            type="number",
                            location="body",
                            required=True,
                            description="Transaction amount"
                        ),
                        RequestParam(
                            name="currency",
                            type="string",
                            location="body",
                            required=True,
                            description="Currency code"
                        )
                    ]
                )
            ]
        )
        
        # Store BAnCS API configuration directly (synchronously)
        self.apis[bancs_config.name] = bancs_config
        
        # Create HTTP client synchronously
        try:
            # We'll create the client later when needed
            logger.info(f"Registered default BAnCS API with {len(bancs_config.endpoints)} endpoints")
        except Exception as e:
            logger.error(f"Error registering default BAnCS API: {e}")
        
        # Load APIs from files
        self._load_apis_from_files()
    
    def _load_apis_from_files(self):
        """Load APIs from specification files."""
        spec_dirs = [Path("api_specs"), Path("collections")]
        
        for spec_dir in spec_dirs:
            if not spec_dir.exists():
                continue
            
            for file_path in spec_dir.glob("*.json"):
                try:
                    with open(file_path) as f:
                        data = json.load(f)
                        
                        # Check if it's a Postman collection
                        if "info" in data and "_postman_id" in data.get("info", {}):
                            self._load_postman_collection(data, file_path)
                        # Check if it's an OpenAPI spec
                        elif "openapi" in data or "swagger" in data:
                            self._load_openapi_spec(data, file_path)
                    
                    logger.info(f"Loaded API spec from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
            
            # Also check for YAML files
            for file_path in spec_dir.glob("*.yaml"):
                try:
                    with open(file_path) as f:
                        data = yaml.safe_load(f)
                        if "openapi" in data or "swagger" in data:
                            self._load_openapi_spec(data, file_path)
                    logger.info(f"Loaded API spec from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
    
    def _setup_core_tools(self):
        """Set up the core management tools."""
        
        @self.mcp.tool()
        async def register_api(config: Dict[str, Any], ctx: Context) -> str:
            """
            Register a new API configuration and create MCP tools for its endpoints.
            
            Args:
                config: API configuration dictionary containing:
                    - name: API name
                    - base_url: Base URL for the API
                    - description: Optional API description
                    - auth: Optional authentication configuration
                    - headers: Optional global headers
                    - endpoints: List of endpoint configurations
            
            Returns:
                Success message with list of created tools
            """
            try:
                api_config = APIConfig(**config)
                
                # Store API configuration
                self.apis[api_config.name] = api_config
                
                # Create HTTP client for this API
                client = await self._create_http_client(api_config)
                self.http_clients[api_config.name] = client
                
                # Create tools for each endpoint
                created_tools = []
                for endpoint in api_config.endpoints:
                    tool_name = f"{api_config.name}_{endpoint.name}"
                    try:
                        await self._create_endpoint_tool(api_config, endpoint, tool_name)
                        created_tools.append(tool_name)
                    except Exception as e:
                        await ctx.error(f"Failed to create tool {tool_name}: {str(e)}")
                        continue
                
                await ctx.info(f"Registered API '{api_config.name}' with {len(created_tools)} tools")
                return f"Successfully registered API '{api_config.name}' with tools: {', '.join(created_tools)}"
                
            except Exception as e:
                await ctx.error(f"Failed to register API: {str(e)}")
                raise
        
        @self.mcp.tool()
        async def list_apis(ctx: Context) -> Dict[str, Any]:
            """
            List all registered APIs and their endpoints.
            
            Returns:
                Dictionary of registered APIs with their configurations
            """
            result = {}
            for name, api in self.apis.items():
                result[name] = {
                    "base_url": api.base_url,
                    "description": api.description,
                    "auth_type": api.auth.type if api.auth else "none",
                    "endpoints": [
                        {
                            "name": ep.name,
                            "method": ep.method,
                            "path": ep.path,
                            "description": ep.description,
                            "parameters": [
                                {
                                    "name": param.name,
                                    "type": param.type,
                                    "location": param.location,
                                    "required": param.required,
                                    "description": param.description,
                                    "default": param.default
                                }
                                for param in ep.params
                            ]
                        }
                        for ep in api.endpoints
                    ]
                }
            return result
        
        @self.mcp.tool()
        async def unregister_api(api_name: str, ctx: Context) -> str:
            """
            Unregister an API and remove its tools.
            
            Args:
                api_name: Name of the API to unregister
            
            Returns:
                Success message
            """
            if api_name not in self.apis:
                raise ValueError(f"API '{api_name}' not found")
            
            api_config = self.apis[api_name]
            
            # Remove tools
            for endpoint in api_config.endpoints:
                tool_name = f"{api_name}_{endpoint.name}"
                try:
                    self.mcp.remove_tool(tool_name)
                except:
                    pass  # Tool might not exist
            
            # Close HTTP client
            if api_name in self.http_clients:
                await self.http_clients[api_name].aclose()
                del self.http_clients[api_name]
            
            # Remove API config
            del self.apis[api_name]
            
            await ctx.info(f"Unregistered API '{api_name}'")
            return f"Successfully unregistered API '{api_name}'"
        
        @self.mcp.tool()
        async def test_api_connection(api_name: str, ctx: Context) -> Dict[str, Any]:
            """
            Test connection to a registered API.
            
            Args:
                api_name: Name of the API to test
            
            Returns:
                Connection test results
            """
            if api_name not in self.apis:
                raise ValueError(f"API '{api_name}' not found")
            
            api_config = self.apis[api_name]
            client = self.http_clients.get(api_name)
            
            if not client:
                raise ValueError(f"No HTTP client found for API '{api_name}'")
            
            try:
                # Try a simple HEAD or GET request to base URL
                response = await client.head(api_config.base_url, timeout=5.0)
                return {
                    "status": "connected",
                    "status_code": response.status_code,
                    "headers": dict(response.headers)
                }
            except Exception as e:
                return {
                    "status": "failed",
                    "error": str(e)
                }
        
        @self.mcp.tool()
        async def call_api(
            api_name: str,
            endpoint_name: str,
            parameters: Dict[str, Any] = None,
            ctx: Optional[Context] = None
        ) -> Dict[str, Any]:
            """
            Call any registered API endpoint with dynamic parameters.
            
            This is a generic tool that allows calling any registered API endpoint
            without having to use the specific endpoint tools.
            
            Args:
                api_name: Name of the registered API to call
                endpoint_name: Name of the endpoint within the API
                parameters: Dictionary of parameters to pass to the endpoint
                ctx: Optional context for logging
            
            Returns:
                API response data and metadata
            """
            if parameters is None:
                parameters = {}
            
            # Validate API exists
            if api_name not in self.apis:
                available_apis = list(self.apis.keys())
                error_msg = f"API '{api_name}' not found. Available APIs: {', '.join(available_apis)}"
                if ctx:
                    await ctx.error(error_msg)
                raise ValueError(error_msg)
            
            # Find the endpoint
            api_config = self.apis[api_name]
            endpoint = None
            for ep in api_config.endpoints:
                if ep.name == endpoint_name:
                    endpoint = ep
                    break
            
            if not endpoint:
                available_endpoints = [ep.name for ep in api_config.endpoints]
                error_msg = f"Endpoint '{endpoint_name}' not found in API '{api_name}'. Available endpoints: {', '.join(available_endpoints)}"
                if ctx:
                    await ctx.error(error_msg)
                raise ValueError(error_msg)
            
            if ctx:
                await ctx.info(f"Calling {api_name}.{endpoint_name} with parameters: {parameters}")
            
            try:
                # Call the API
                response_data = await self._execute_api_call(
                    api_name=api_name,
                    endpoint_name=endpoint_name,
                    params=parameters,
                    ctx=ctx
                )
                
                # Return structured response
                result = {
                    "success": True,
                    "api_name": api_name,
                    "endpoint_name": endpoint_name,
                    "endpoint_info": {
                        "method": endpoint.method,
                        "path": endpoint.path,
                        "description": endpoint.description
                    },
                    "parameters_used": parameters,
                    "data": response_data
                }
                
                if ctx:
                    await ctx.info(f"Successfully called {api_name}.{endpoint_name}")
                
                return result
                
            except Exception as e:
                error_result = {
                    "success": False,
                    "api_name": api_name,
                    "endpoint_name": endpoint_name,
                    "parameters_used": parameters,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                
                if ctx:
                    await ctx.error(f"Failed to call {api_name}.{endpoint_name}: {str(e)}")
                
                return error_result
        
        @self.mcp.tool()
        async def get_api_schema(api_name: str, endpoint_name: str = None, ctx: Optional[Context] = None) -> Dict[str, Any]:
            """
            Get the schema/documentation for an API or specific endpoint.
            
            Args:
                api_name: Name of the registered API
                endpoint_name: Optional specific endpoint name
                ctx: Optional context for logging
            
            Returns:
                Schema information for the API or endpoint
            """
            if api_name not in self.apis:
                available_apis = list(self.apis.keys())
                error_msg = f"API '{api_name}' not found. Available APIs: {', '.join(available_apis)}"
                if ctx:
                    await ctx.error(error_msg)
                raise ValueError(error_msg)
            
            api_config = self.apis[api_name]
            
            if endpoint_name:
                # Return specific endpoint schema
                endpoint = None
                for ep in api_config.endpoints:
                    if ep.name == endpoint_name:
                        endpoint = ep
                        break
                
                if not endpoint:
                    available_endpoints = [ep.name for ep in api_config.endpoints]
                    error_msg = f"Endpoint '{endpoint_name}' not found. Available: {', '.join(available_endpoints)}"
                    if ctx:
                        await ctx.error(error_msg)
                    raise ValueError(error_msg)
                
                return {
                    "api_name": api_name,
                    "endpoint_name": endpoint_name,
                    "method": endpoint.method,
                    "path": endpoint.path,
                    "description": endpoint.description,
                    "parameters": [
                        {
                            "name": param.name,
                            "type": param.type,
                            "location": param.location,
                            "required": param.required,
                            "description": param.description,
                            "default": param.default,
                            "enum": param.enum
                        }
                        for param in endpoint.params
                    ],
                    "headers": endpoint.headers,
                    "timeout": endpoint.timeout
                }
            else:
                # Return all endpoints schema
                return {
                    "api_name": api_name,
                    "base_url": api_config.base_url,
                    "description": api_config.description,
                    "auth_type": api_config.auth.type if api_config.auth else "none",
                    "global_headers": api_config.headers,
                    "endpoints": [
                        {
                            "name": ep.name,
                            "method": ep.method,
                            "path": ep.path,
                            "description": ep.description,
                            "parameters": [
                                {
                                    "name": param.name,
                                    "type": param.type,
                                    "location": param.location,
                                    "required": param.required,
                                    "description": param.description,
                                    "default": param.default,
                                    "enum": param.enum
                                }
                                for param in ep.params
                            ]
                        }
                        for ep in api_config.endpoints
                    ]
                }
            """
            Register a new API configuration and create MCP tools for its endpoints.
            
            Args:
                config: API configuration dictionary containing:
                    - name: API name
                    - base_url: Base URL for the API
                    - description: Optional API description
                    - auth: Optional authentication configuration
                    - headers: Optional global headers
                    - endpoints: List of endpoint configurations
            
            Returns:
                Success message with list of created tools
            """
            try:
                api_config = APIConfig(**config)
                
                # Store API configuration
                self.apis[api_config.name] = api_config
                
                # Create HTTP client for this API
                client = await self._create_http_client(api_config)
                self.http_clients[api_config.name] = client
                
                # Create tools for each endpoint
                created_tools = []
                for endpoint in api_config.endpoints:
                    tool_name = f"{api_config.name}_{endpoint.name}"
                    try:
                        await self._create_endpoint_tool(api_config, endpoint, tool_name)
                        created_tools.append(tool_name)
                    except Exception as e:
                        await ctx.error(f"Failed to create tool {tool_name}: {str(e)}")
                        continue
                
                await ctx.info(f"Registered API '{api_config.name}' with {len(created_tools)} tools")
                return f"Successfully registered API '{api_config.name}' with tools: {', '.join(created_tools)}"
                
            except Exception as e:
                await ctx.error(f"Failed to register API: {str(e)}")
                raise
        
        @self.mcp.tool()
        async def list_apis(ctx: Context) -> Dict[str, Any]:
            """
            List all registered APIs and their endpoints.
            
            Returns:
                Dictionary of registered APIs with their configurations
            """
            result = {}
            for name, api in self.apis.items():
                result[name] = {
                    "base_url": api.base_url,
                    "description": api.description,
                    "auth_type": api.auth.type if api.auth else "none",
                    "endpoints": [
                        {
                            "name": ep.name,
                            "method": ep.method,
                            "path": ep.path,
                            "description": ep.description,
                            "parameters": [
                                {
                                    "name": param.name,
                                    "type": param.type,
                                    "location": param.location,
                                    "required": param.required,
                                    "description": param.description,
                                    "default": param.default
                                }
                                for param in ep.params
                            ]
                        }
                        for ep in api.endpoints
                    ]
                }
            return result
        
        @self.mcp.tool()
        async def unregister_api(api_name: str, ctx: Context) -> str:
            """
            Unregister an API and remove its tools.
            
            Args:
                api_name: Name of the API to unregister
                ctx: MCP context
            
            Returns:
                Success message
            """
            if api_name not in self.apis:
                raise ValueError(f"API '{api_name}' not found")
            
            api_config = self.apis[api_name]
            
            # Remove tools
            for endpoint in api_config.endpoints:
                tool_name = f"{api_name}_{endpoint.name}"
                try:
                    if self.mcp:
                        self.mcp.remove_tool(tool_name)
                except:
                    pass  # Tool might not exist
            
            # Close HTTP client
            if api_name in self.http_clients:
                await self.http_clients[api_name].aclose()
                del self.http_clients[api_name]
            
            # Remove API config
            del self.apis[api_name]
            
            await ctx.info(f"Unregistered API '{api_name}'")
            return f"Successfully unregistered API '{api_name}'"
        
        # Add to MCP if available
        if self.mcp :
            unregister_api.__name__ = "unregister_api"
            self.mcp.add_tool(unregister_api)
        
        # Define test_api_connection
        async def test_api_connection(api_name: str, ctx: Context) -> Dict[str, Any]:
            """
            Test connection to a registered API.
            
            Args:
                api_name: Name of the API to test
                ctx: MCP context
            
            Returns:
                Connection test results
            """
            if api_name not in self.apis:
                raise ValueError(f"API '{api_name}' not found")
            
            api_config = self.apis[api_name]
            client = self.http_clients.get(api_name)
            
            if not client:
                # Create client if needed
                client = await self._create_http_client(api_config)
                self.http_clients[api_name] = client
            
            try:
                # Try a simple HEAD or GET request to base URL
                response = await client.head(api_config.base_url, timeout=5.0)
                return {
                    "status": "connected",
                    "status_code": response.status_code,
                    "headers": dict(response.headers)
                }
            except Exception as e:
                return {
                    "status": "failed",
                    "error": str(e)
                }
        
        # Add to MCP if available
        if self.mcp :
            test_api_connection.__name__ = "test_api_connection"
            self.mcp.add_tool(test_api_connection)
        
        # Define call_api
        async def call_api(
            api_name: str,
            endpoint_name: str,
            parameters: Dict[str, Any] = None,
            ctx: Optional[Context] = None
        ) -> Dict[str, Any]:
            """
            Call any registered API endpoint with dynamic parameters.
            
            Args:
                api_name: Name of the registered API to call
                endpoint_name: Name of the endpoint within the API
                parameters: Dictionary of parameters to pass to the endpoint
                ctx: Optional context for logging
            
            Returns:
                API response data and metadata
            """
            if parameters is None:
                parameters = {}
            
            # Validate API exists
            if api_name not in self.apis:
                available_apis = list(self.apis.keys())
                error_msg = f"API '{api_name}' not found. Available APIs: {', '.join(available_apis)}"
                if ctx:
                    await ctx.error(error_msg)
                raise ValueError(error_msg)
            
            # Find the endpoint
            api_config = self.apis[api_name]
            endpoint = None
            for ep in api_config.endpoints:
                if ep.name == endpoint_name:
                    endpoint = ep
                    break
            
            if not endpoint:
                available_endpoints = [ep.name for ep in api_config.endpoints]
                error_msg = f"Endpoint '{endpoint_name}' not found in API '{api_name}'. Available endpoints: {', '.join(available_endpoints)}"
                if ctx:
                    await ctx.error(error_msg)
                raise ValueError(error_msg)
            
            if ctx:
                await ctx.info(f"Calling {api_name}.{endpoint_name} with parameters: {parameters}")
            
            try:
                # Call the API using existing method
                response_data = await self._execute_api_call(
                    api_name=api_name,
                    endpoint_name=endpoint_name,
                    params=parameters,
                    ctx=ctx
                )
                
                # Return structured response with metadata
                result = {
                    "success": True,
                    "api_name": api_name,
                    "endpoint_name": endpoint_name,
                    "endpoint_info": {
                        "method": endpoint.method,
                        "path": endpoint.path,
                        "description": endpoint.description
                    },
                    "parameters_used": parameters,
                    "data": response_data
                }
                
                if ctx:
                    await ctx.info(f"Successfully called {api_name}.{endpoint_name}")
                
                return result
                
            except Exception as e:
                error_result = {
                    "success": False,
                    "api_name": api_name,
                    "endpoint_name": endpoint_name,
                    "parameters_used": parameters,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                
                if ctx:
                    await ctx.error(f"Failed to call {api_name}.{endpoint_name}: {str(e)}")
                
                return error_result
        
        # Store for internal use
        self.call_api = call_api
        
        # Add to MCP if available
        if self.mcp :
            call_api.__name__ = "call_api"
            self.mcp.add_tool(call_api)
        
        # Define get_api_schema
        async def get_api_schema(api_name: str, endpoint_name: str = None, ctx: Optional[Context] = None) -> Dict[str, Any]:
            """
            Get the schema/documentation for an API or specific endpoint.
            
            Args:
                api_name: Name of the registered API
                endpoint_name: Optional specific endpoint name
                ctx: Optional context for logging
            
            Returns:
                Schema information for the API or endpoint
            """
            if api_name not in self.apis:
                available_apis = list(self.apis.keys())
                error_msg = f"API '{api_name}' not found. Available APIs: {', '.join(available_apis)}"
                if ctx:
                    await ctx.error(error_msg)
                raise ValueError(error_msg)
            
            api_config = self.apis[api_name]
            
            if endpoint_name:
                # Return specific endpoint schema
                endpoint = None
                for ep in api_config.endpoints:
                    if ep.name == endpoint_name:
                        endpoint = ep
                        break
                
                if not endpoint:
                    available_endpoints = [ep.name for ep in api_config.endpoints]
                    error_msg = f"Endpoint '{endpoint_name}' not found. Available: {', '.join(available_endpoints)}"
                    if ctx:
                        await ctx.error(error_msg)
                    raise ValueError(error_msg)
                
                return {
                    "api_name": api_name,
                    "endpoint_name": endpoint_name,
                    "method": endpoint.method,
                    "path": endpoint.path,
                    "description": endpoint.description,
                    "parameters": [
                        {
                            "name": param.name,
                            "type": param.type,
                            "location": param.location,
                            "required": param.required,
                            "description": param.description,
                            "default": param.default,
                            "enum": param.enum
                        }
                        for param in endpoint.params
                    ],
                    "headers": endpoint.headers,
                    "timeout": endpoint.timeout
                }
            else:
                # Return all endpoints schema
                return {
                    "api_name": api_name,
                    "base_url": api_config.base_url,
                    "description": api_config.description,
                    "auth_type": api_config.auth.type if api_config.auth else "none",
                    "global_headers": api_config.headers,
                    "endpoints": [
                        {
                            "name": ep.name,
                            "method": ep.method,
                            "path": ep.path,
                            "description": ep.description,
                            "parameters": [
                                {
                                    "name": param.name,
                                    "type": param.type,
                                    "location": param.location,
                                    "required": param.required,
                                    "description": param.description,
                                    "default": param.default,
                                    "enum": param.enum
                                }
                                for param in ep.params
                            ]
                        }
                        for ep in api_config.endpoints
                    ]
                }
        
        # Add to MCP if available
        if self.mcp :
            get_api_schema.__name__ = "get_api_schema"
            self.mcp.add_tool(get_api_schema)
        
        # Store register_api for internal use
        self.register_api = register_api
    
    async def _register_api_internal(self, config: Dict[str, Any]):
        """Internal method to register an API without context."""
        # if not FASTMCP_AVAILABLE:
        #     logger.warning("Cannot register API - MCP not available")
        #     return
        
        try:
            # Create a dummy context for internal use
            class DummyContext:
                async def info(self, msg): logger.info(msg)
                async def error(self, msg): logger.error(msg)
            
            ctx = DummyContext()
            # Call the implementation directly, not through MCP wrapper
            result = await self.register_api(config, ctx)
            logger.info(f"Internal registration: {result}")
        except Exception as e:
            logger.error(f"Error in internal API registration: {e}")
    
    async def _create_http_client(self, api_config: APIConfig) -> httpx.AsyncClient:
        """Create an HTTP client with authentication configured."""
        headers = {}
        auth = None
        
        # Add global headers
        if api_config.headers:
            headers.update(api_config.headers)
        
        # Configure authentication
        if api_config.auth:
            auth_config = api_config.auth
            
            if auth_config.type == "bearer" and auth_config.bearer_token:
                headers["Authorization"] = f"Bearer {auth_config.bearer_token}"
            
            elif auth_config.type == "api_key":
                if auth_config.api_key_header and auth_config.api_key:
                    headers[auth_config.api_key_header] = auth_config.api_key
            
            elif auth_config.type == "basic" and auth_config.username and auth_config.password:
                auth = httpx.BasicAuth(auth_config.username, auth_config.password)
            
            elif auth_config.type == "custom" and auth_config.custom_headers:
                headers.update(auth_config.custom_headers)
        
        # Create client
        client = httpx.AsyncClient(
            base_url=api_config.base_url,
            headers=headers,
            auth=auth,
            timeout=30.0,
            follow_redirects=True
        )
        
        return client
    
    async def _create_endpoint_tool(self, api_config: APIConfig, endpoint: APIEndpoint, tool_name: str):
        """Create an MCP tool for a specific API endpoint using closure approach."""
        
        # Build parameter signature
        sig_parts = []
        param_annotations = {}
        
        for param in endpoint.params:
            # Determine Python type
            param_type = str
            if param.type == "integer":
                param_type = int
            elif param.type == "number":
                param_type = float
            elif param.type == "boolean":
                param_type = bool
            elif param.type == "array":
                param_type = List[Any]
            elif param.type == "object":
                param_type = Dict[str, Any]
            
            # Build parameter with default if not required
            if param.required:
                sig_parts.append(param.name)
            else:
                default_val = param.default if param.default is not None else None
                sig_parts.append(f"{param.name}={repr(default_val)}")
            
            param_annotations[param.name] = param_type
        
        # Add context parameter
        sig_parts.append("ctx: Optional[Context] = None")
        param_annotations["ctx"] = Optional[Context]
        param_annotations["return"] = Any
        
        # Create the closure-based tool function
        def create_tool_function():
            # Capture the current values
            api_name = api_config.name
            endpoint_name = endpoint.name
            
            async def api_tool_func(*args, **kwargs):
                # Map positional args to parameter names
                param_names = [p.name for p in endpoint.params]
                call_params = {}
                
                # Handle positional arguments
                for i, arg in enumerate(args):
                    if i < len(param_names):
                        call_params[param_names[i]] = arg
                
                # Handle keyword arguments
                ctx = kwargs.pop('ctx', None)
                call_params.update(kwargs)
                
                return await self._execute_api_call(
                    api_name=api_name,
                    endpoint_name=endpoint_name,
                    params=call_params,
                    ctx=ctx
                )
            
            # Set function metadata
            api_tool_func.__name__ = tool_name
            api_tool_func.__doc__ = f"""
{endpoint.description}

Generated tool for {api_config.name} API endpoint: {endpoint.method} {endpoint.path}

Parameters:
{self._generate_param_docs(endpoint)}
"""
            api_tool_func.__annotations__ = param_annotations
            
            return api_tool_func
        
        # Create and register the tool
        tool_function = create_tool_function()
        self.mcp.add_tool(tool_function)
    
    def _generate_param_docs(self, endpoint: APIEndpoint) -> str:
        """Generate parameter documentation for the tool."""
        docs = []
        for param in endpoint.params:
            required_str = "required" if param.required else "optional"
            default_str = f" (default: {param.default})" if param.default is not None else ""
            desc = param.description or ""
            docs.append(f"- {param.name} ({param.type}, {required_str}){default_str}: {desc}")
        return "\n".join(docs)
    
    async def _execute_api_call(self, api_name: str, endpoint_name: str, params: Dict[str, Any], ctx: Optional[Context] = None) -> Any:
        """Execute an API call with the given parameters."""
        
        # Get API config and endpoint
        api_config = self.apis.get(api_name)
        if not api_config:
            raise ValueError(f"API '{api_name}' not found")
        
        endpoint = None
        for ep in api_config.endpoints:
            if ep.name == endpoint_name:
                endpoint = ep
                break
        
        if not endpoint:
            raise ValueError(f"Endpoint '{endpoint_name}' not found in API '{api_name}'")
        
        # Get or create HTTP client
        client = self.http_clients.get(api_name)
        if not client:
            # Create client on demand
            client = await self._create_http_client(api_config)
            self.http_clients[api_name] = client
        
        # Build request
        url_path = endpoint.path
        query_params = {}
        headers = {}
        json_body = None
        
        # Add endpoint-specific headers
        if endpoint.headers:
            headers.update(endpoint.headers)
        
        # Process parameters
        for param in endpoint.params:
            value = params.get(param.name)
            
            # Use default value if not provided
            if value is None and param.default is not None:
                value = param.default
            
            # Check required parameters
            if value is None and param.required:
                raise ValueError(f"Required parameter '{param.name}' not provided")
            
            # Skip None values for optional parameters
            if value is None:
                continue
            
            if param.location == "path":
                # Replace path parameter
                url_path = url_path.replace(f"{{{param.name}}}", quote(str(value)))
            elif param.location == "query":
                query_params[param.name] = value
            elif param.location == "header":
                headers[param.name] = str(value)
            elif param.location == "body":
                if json_body is None:
                    json_body = {}
                json_body[param.name] = value
        
        # Handle API key in query params
        if api_config.auth and api_config.auth.type == "api_key" and api_config.auth.api_key_param:
            query_params[api_config.auth.api_key_param] = api_config.auth.api_key
        
        # Make request
        try:
            if ctx:
                await ctx.info(f"Calling {endpoint.method} {url_path}")
            
            response = await client.request(
                method=endpoint.method,
                url=url_path,
                params=query_params if query_params else None,
                headers=headers if headers else None,
                json=json_body,
                timeout=endpoint.timeout
            )
            
            response.raise_for_status()
            
            # Parse response
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                return response.json()
            else:
                return response.text
                
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            if ctx:
                await ctx.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            if ctx:
                await ctx.error(f"Request failed: {str(e)}")
            raise
    
    def _load_postman_collection(self, collection: Dict[str, Any], file_path: Path):
        """Convert Postman collection to API configuration."""
        api_name = collection.get("info", {}).get("name", "Postman_API").replace(" ", "_")
        
        endpoints = []
        self._parse_postman_items(collection.get("item", []), endpoints)
        
        if endpoints:
            config = {
                "name": api_name,
                "base_url": self._extract_base_url_from_postman(collection),
                "description": collection.get("info", {}).get("description", f"Loaded from {file_path.name}"),
                "endpoints": endpoints
            }
            asyncio.create_task(self._register_api_internal(config))
    
    def _extract_base_url_from_postman(self, collection: Dict[str, Any]) -> str:
        """Extract base URL from Postman collection."""
        # Try to find base URL from variables
        for var in collection.get("variable", []):
            if var.get("key") in ["base_url", "baseUrl", "url"]:
                return var.get("value", "https://api.example.com")
        
        # Try to extract from first request
        items = collection.get("item", [])
        if items:
            first_item = items[0]
            if "request" in first_item:
                url = first_item["request"].get("url", {})
                if isinstance(url, dict) and "raw" in url:
                    # Extract base from raw URL
                    raw_url = url["raw"]
                    if raw_url.startswith("http"):
                        parts = raw_url.split("/")
                        if len(parts) >= 3:
                            return "/".join(parts[:3])
        
        return "https://api.example.com"
    
    def _parse_postman_items(self, items: List[Dict], endpoints: List[Dict], prefix: str = ""):
        """Recursively parse Postman collection items."""
        for item in items:
            if "item" in item:
                # It's a folder, recurse
                folder_name = item.get("name", "")
                self._parse_postman_items(item["item"], endpoints, f"{prefix}{folder_name}_")
            elif "request" in item:
                # It's an endpoint
                request = item["request"]
                endpoint_name = f"{prefix}{item.get('name', 'endpoint')}".replace(" ", "_").lower()
                
                endpoint = {
                    "name": endpoint_name,
                    "method": request.get("method", "GET"),
                    "path": self._extract_postman_path(request),
                    "description": item.get("description", ""),
                    "params": self._extract_postman_params(request)
                }
                endpoints.append(endpoint)
    
    def _extract_postman_path(self, request: Dict) -> str:
        """Extract path from Postman request."""
        url = request.get("url", {})
        if isinstance(url, str):
            # Simple string URL - extract path
            if url.startswith("http"):
                parts = url.split("/", 3)
                return "/" + parts[3] if len(parts) > 3 else "/"
            return url
        elif isinstance(url, dict):
            # Structured URL object
            path = url.get("path", [])
            if isinstance(path, list):
                # Replace variables with placeholders
                path_parts = []
                for part in path:
                    if part.startswith(":"):
                        # It's a variable
                        path_parts.append(f"{{{part[1:]}}}")
                    else:
                        path_parts.append(part)
                return "/" + "/".join(path_parts)
            
            # Try raw URL
            if "raw" in url:
                raw = url["raw"]
                if raw.startswith("http"):
                    parts = raw.split("/", 3)
                    return "/" + parts[3] if len(parts) > 3 else "/"
        return "/"
    
    def _extract_postman_params(self, request: Dict) -> List[Dict]:
        """Extract parameters from Postman request."""
        params = []
        url = request.get("url", {})
        
        if isinstance(url, dict):
            # Query parameters
            for param in url.get("query", []):
                params.append({
                    "name": param.get("key", ""),
                    "type": "string",
                    "location": "query",
                    "required": not param.get("disabled", False),
                    "description": param.get("description", ""),
                    "default": param.get("value") if param.get("disabled") else None
                })
            
            # Path variables
            for var in url.get("variable", []):
                params.append({
                    "name": var.get("key", ""),
                    "type": "string",
                    "location": "path",
                    "required": True,
                    "description": var.get("description", "")
                })
        
        # Body parameters (if any)
        body = request.get("body", {})
        if body.get("mode") == "raw":
            try:
                # Try to parse JSON body
                raw_body = json.loads(body.get("raw", "{}"))
                for key in raw_body.keys():
                    params.append({
                        "name": key,
                        "type": "string",
                        "location": "body",
                        "required": False,
                        "description": ""
                    })
            except:
                pass
        
        return params
    
    def _load_openapi_spec(self, spec: Dict[str, Any], file_path: Path):
        """Convert OpenAPI spec to API configuration."""
        api_name = spec.get("info", {}).get("title", "OpenAPI").replace(" ", "_")
        base_url = ""
        
        # Extract base URL from servers
        if "servers" in spec and spec["servers"]:
            base_url = spec["servers"][0].get("url", "")
        
        endpoints = []
        
        for path, methods in spec.get("paths", {}).items():
            for method, details in methods.items():
                if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                    endpoint = {
                        "name": details.get("operationId", f"{method}_{path}").replace("/", "_").lower(),
                        "method": method.upper(),
                        "path": path,
                        "description": details.get("summary", details.get("description", "")),
                        "params": self._extract_openapi_params(details)
                    }
                    endpoints.append(endpoint)
        
        if endpoints:
            config = {
                "name": api_name,
                "base_url": base_url or "https://api.example.com",
                "description": spec.get("info", {}).get("description", f"Loaded from {file_path.name}"),
                "endpoints": endpoints
            }
            asyncio.create_task(self._register_api_internal(config))
    
    def _extract_openapi_params(self, operation: Dict) -> List[Dict]:
        """Extract parameters from OpenAPI operation."""
        params = []
        
        for param in operation.get("parameters", []):
            params.append({
                "name": param.get("name", ""),
                "type": param.get("schema", {}).get("type", "string"),
                "location": param.get("in", "query"),
                "required": param.get("required", False),
                "description": param.get("description", ""),
                "default": param.get("schema", {}).get("default"),
                "enum": param.get("schema", {}).get("enum")
            })
        
        # Handle request body
        if "requestBody" in operation:
            content = operation["requestBody"].get("content", {})
            if "application/json" in content:
                schema = content["application/json"].get("schema", {})
                if "properties" in schema:
                    for prop_name, prop_schema in schema["properties"].items():
                        params.append({
                            "name": prop_name,
                            "type": prop_schema.get("type", "string"),
                            "location": "body",
                            "required": prop_name in schema.get("required", []),
                            "description": prop_schema.get("description", "")
                        })
        
        return params
    
    def run(self, **kwargs):
        """Run the MCP server."""
        if self.mcp:
            return self.mcp.run(**kwargs)
        else:
            logger.warning("Cannot run MCP server - FastMCP not available")
            return None

# Maintain backward compatibility
APIWeaver = BAnCSAPIWeaver