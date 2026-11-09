# backend/apiweaver.py
"""
Advanced APIWeaver implementation with full MCP support
Complete replacement with all features from the advanced version
"""

import json
import asyncio
import secrets
import time
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

# Writes from MCP clients wait for a human (docs/decode-fork-plan.md: MCP endpoint, decided 2026-11-09):
# an MCP client has no Approve button, so call_api returns a link the user opens on the site.
WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
HOLD_TTL_S = 600
APPROVE_URL = os.getenv("MCP_APPROVE_URL", "https://apicookbook.vercel.app/#approve-")

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
    example: Optional[Any] = None  # Add example field

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

class BaNCSAPIWeaver:
    """Main server that creates MCP tools from API configurations."""
    
    def __init__(self, name: str = "TCS-BaNCS-APIWeaver"):
        self.mcp = FastMCP(name)
        self.apis: Dict[str, APIConfig] = {}
        self.http_clients: Dict[str, httpx.AsyncClient] = {}
        self.created_tools: Dict[str, Any] = {}  # Track created tools
        self.pending_registrations: List[Dict[str, Any]] = []  # Store configs to register later
        self.held_writes: Dict[str, Dict[str, Any]] = {}  # approval id -> the write, its status and result
        self._setup_core_tools()
        self._initialize_default_apis()
    
    def _initialize_default_apis(self):
        """Initialize with default BaNCS APIs and load from files."""
        # Default BaNCS configuration
        bancs_config = APIConfig(
            name="TCS_BaNCS",
            base_url="https://demoapps.tcsbancs.com",
            description="TCS BaNCS Banking APIs",
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
                            description="15-digit account reference number",
                            example="101000000101814"
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
                            description="Customer ID",
                            example="146025"
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
                            description="Customer type (1=Individual, 2=Corporate)",
                            default=1
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
                            description="Nationality code",
                            default="US"
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
                            description="Currency code",
                            default="USD"
                        )
                    ]
                )
            ]
        )
        
        # Store BaNCS API configuration directly (synchronously)
        self.apis[bancs_config.name] = bancs_config
        logger.info(f"Registered default BaNCS API with {len(bancs_config.endpoints)} endpoints")
        
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
                    "endpoints_count": len(api.endpoints),
                    "endpoints": [
                        {
                            "name": ep.name,
                            "method": ep.method,
                            "path": ep.path,
                            "description": ep.description
                        }
                        for ep in api.endpoints[:10]  # Show first 10 endpoints
                    ]
                }
            return result
        
        @self.mcp.tool()
        async def call_api(
            api_name: str,
            endpoint_name: str,
            parameters: Optional[Dict[str, Any]] = None,
            ctx: Optional[Context] = None
        ) -> Dict[str, Any]:
            """
            Call any registered API endpoint with dynamic parameters.
            
            Use get_api_schema first to discover required parameters.
            
            Args:
                api_name: Name of the registered API to call
                endpoint_name: Name of the endpoint within the API
                parameters: Dictionary of parameters to pass to the endpoint
                ctx: Optional context for logging
            
            Returns:
                API response data and metadata
            
            Example:
                # First check schema
                schema = await get_api_schema("TCS_BaNCS", "get_account_balance")
                
                # Then call with correct parameters
                result = await call_api(
                    api_name="TCS_BaNCS",
                    endpoint_name="get_account_balance",
                    parameters={"accountReference": "101000000101814"}
                )
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
                available_endpoints = [ep.name for ep in api_config.endpoints][:10]
                error_msg = f"Endpoint '{endpoint_name}' not found in API '{api_name}'. Available endpoints include: {', '.join(available_endpoints)}..."
                if ctx:
                    await ctx.error(error_msg)
                raise ValueError(error_msg)
            
            # A write is held for a human; the same call with {"approval_id": ...} returns its outcome.
            approval_id = parameters.pop("approval_id", None)
            if approval_id:
                return self._held_outcome(str(approval_id))
            if endpoint.method.upper() in WRITE_METHODS:
                return self._hold_write(api_name, endpoint, parameters)

            if ctx:
                await ctx.info(f"Calling {api_name}.{endpoint_name}")
            
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
                    "data": response_data
                }
                
                if ctx:
                    await ctx.info(f"Successfully called {api_name}.{endpoint_name}")
                
                return result
                
            except Exception as e:
                # Check if it's a missing parameter error
                if "Required parameter" in str(e):
                    # Provide helpful error with example
                    required_params = [p.name for p in endpoint.params if p.required]
                    example_params = {}
                    for param in endpoint.params:
                        if param.required:
                            if param.example:
                                example_params[param.name] = param.example
                            elif param.default:
                                example_params[param.name] = param.default
                            else:
                                example_params[param.name] = f"<{param.type}>"
                    
                    error_msg = f"""
Missing required parameter for {api_name}.{endpoint_name}

Required parameters: {required_params}

Example call:
await call_api(
    api_name="{api_name}",
    endpoint_name="{endpoint_name}",
    parameters={json.dumps(example_params, indent=4)}
)

Use get_api_schema("{api_name}", "{endpoint_name}") for full parameter details.
"""
                    raise ValueError(error_msg)
                
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
        async def get_api_schema(
            api_name: str, 
            endpoint_name: Optional[str] = None, 
            ctx: Optional[Context] = None
        ) -> Dict[str, Any]:
            """
            Get the schema/documentation for an API or specific endpoint.
            Shows required parameters, types, and examples.
            
            Args:
                api_name: Name of the registered API
                endpoint_name: Optional specific endpoint name
                ctx: Optional context for logging
            
            Returns:
                Schema information with parameter details and examples
            """
            if api_name not in self.apis:
                available_apis = list(self.apis.keys())
                error_msg = f"API '{api_name}' not found. Available APIs: {', '.join(available_apis)}"
                if ctx:
                    await ctx.error(error_msg)
                raise ValueError(error_msg)
            
            api_config = self.apis[api_name]
            
            if endpoint_name:
                # Return specific endpoint schema with example
                endpoint = None
                for ep in api_config.endpoints:
                    if ep.name == endpoint_name:
                        endpoint = ep
                        break
                
                if not endpoint:
                    available_endpoints = [ep.name for ep in api_config.endpoints][:10]
                    error_msg = f"Endpoint '{endpoint_name}' not found. Available: {', '.join(available_endpoints)}..."
                    if ctx:
                        await ctx.error(error_msg)
                    raise ValueError(error_msg)
                
                # Build example parameters
                example_params = {}
                for param in endpoint.params:
                    if param.example:
                        example_params[param.name] = param.example
                    elif param.default:
                        example_params[param.name] = param.default
                    elif param.required:
                        example_params[param.name] = f"<required_{param.type}>"
                
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
                            "example": param.example,
                            "enum": param.enum
                        }
                        for param in endpoint.params
                    ],
                    "example_call": {
                        "tool": "call_api",
                        "parameters": {
                            "api_name": api_name,
                            "endpoint_name": endpoint_name,
                            "parameters": example_params
                        }
                    }
                }
            else:
                # Return API overview
                return {
                    "api_name": api_name,
                    "base_url": api_config.base_url,
                    "description": api_config.description,
                    "auth_type": api_config.auth.type if api_config.auth else "none",
                    "total_endpoints": len(api_config.endpoints),
                    "endpoints": [
                        {
                            "name": ep.name,
                            "method": ep.method,
                            "path": ep.path,
                            "description": ep.description
                        }
                        for ep in api_config.endpoints[:20]  # Show first 20
                    ],
                    "note": f"Use get_api_schema('{api_name}', 'endpoint_name') for specific endpoint details"
                }
        
        @self.mcp.tool()
        async def search_endpoints(
            query: str,
            api_name: Optional[str] = None,
            method: Optional[str] = None,
            ctx: Optional[Context] = None
        ) -> List[Dict[str, Any]]:
            """
            Search for endpoints across registered APIs.
            
            Args:
                query: Search term to look for in endpoint names, paths, or descriptions
                api_name: Optional - limit search to specific API
                method: Optional - filter by HTTP method (GET, POST, etc.)
                ctx: Optional context for logging
            
            Returns:
                List of matching endpoints with their details
            """
            results = []
            
            for name, api in self.apis.items():
                if api_name and name != api_name:
                    continue
                
                for endpoint in api.endpoints:
                    if method and endpoint.method != method:
                        continue
                    
                    # Search in name, path, and description
                    search_text = query.lower()
                    if (search_text in endpoint.name.lower() or
                        search_text in endpoint.path.lower() or
                        (endpoint.description and search_text in endpoint.description.lower())):
                        
                        results.append({
                            "api": name,
                            "endpoint": endpoint.name,
                            "method": endpoint.method,
                            "path": endpoint.path,
                            "description": endpoint.description,
                            "usage": f"call_api('{name}', '{endpoint.name}', parameters={{...}})"
                        })
            
            if ctx:
                await ctx.info(f"Found {len(results)} endpoints matching '{query}'")
            
            return results
    
    async def _register_api_internal(self, config: Dict[str, Any]):
        """Internal method to register an API without creating individual tools."""
        try:
            api_config = APIConfig(**config)
            
            # Store API configuration directly
            self.apis[api_config.name] = api_config
            
            # Create HTTP client for this API
            client = await self._create_http_client(api_config)
            self.http_clients[api_config.name] = client
            
            # Don't create individual tools - use call_api for all endpoints
            logger.info(f"Registered API '{api_config.name}' with {len(api_config.endpoints)} endpoints (use call_api tool)")
            
        except Exception as e:
            logger.error(f"Error in internal API registration: {e}")
    
    async def process_pending_registrations(self):
        """Process any pending API registrations."""
        if not self.pending_registrations:
            return
            
        for config in self.pending_registrations:
            await self._register_api_internal(config)
        
        count = len(self.pending_registrations)
        self.pending_registrations.clear()
        logger.info(f"Processed {count} pending registrations")
    
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
            
            if auth_config.type == AuthType.BEARER and auth_config.bearer_token:
                headers["Authorization"] = f"Bearer {auth_config.bearer_token}"
            
            elif auth_config.type == AuthType.API_KEY:
                if auth_config.api_key_header and auth_config.api_key:
                    headers[auth_config.api_key_header] = auth_config.api_key
            
            elif auth_config.type == AuthType.BASIC and auth_config.username and auth_config.password:
                auth = httpx.BasicAuth(auth_config.username, auth_config.password)
            
            elif auth_config.type == AuthType.CUSTOM and auth_config.custom_headers:
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
    
    def _generate_param_docs(self, endpoint: APIEndpoint) -> str:
        """Generate parameter documentation for the tool."""
        docs = []
        for param in endpoint.params:
            required_str = "required" if param.required else "optional"
            default_str = f" (default: {param.default})" if param.default is not None else ""
            example_str = f" (example: {param.example})" if param.example else ""
            desc = param.description or ""
            docs.append(f"- {param.name} ({param.type}, {required_str}){default_str}{example_str}: {desc}")
        return "\n".join(docs) if docs else "No parameters"
    
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
        if api_config.auth and api_config.auth.type == AuthType.API_KEY and api_config.auth.api_key_param:
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
        """Convert Postman collection to API configuration with examples."""
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
            # Queue for later registration
            self.pending_registrations.append(config)
            logger.info(f"Queued Postman collection {api_name} for registration ({len(endpoints)} endpoints)")
    
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
        """Extract parameters from Postman request with examples."""
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
                    "default": param.get("value") if param.get("value") else None,
                    "example": param.get("value")  # Use value as example
                })
            
            # Path variables
            for var in url.get("variable", []):
                params.append({
                    "name": var.get("key", ""),
                    "type": "string",
                    "location": "path",
                    "required": True,
                    "description": var.get("description", ""),
                    "example": var.get("value")  # Use value as example
                })
        
        # Body parameters
        body = request.get("body", {})
        if body.get("mode") == "raw":
            try:
                raw_body = json.loads(body.get("raw", "{}"))
                for key, value in raw_body.items():
                    params.append({
                        "name": key,
                        "type": self._infer_type(value),
                        "location": "body",
                        "required": False,
                        "description": "",
                        "default": value,
                        "example": value  # Use value as example
                    })
            except:
                pass
        
        return params
    
    def _infer_type(self, value):
        """Infer parameter type from value."""
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        else:
            return "string"
    
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
            # Queue for later registration
            self.pending_registrations.append(config)
            logger.info(f"Queued OpenAPI spec {api_name} for registration ({len(endpoints)} endpoints)")
    
    def _extract_openapi_params(self, operation: Dict) -> List[Dict]:
        """Extract parameters from OpenAPI operation."""
        params = []
        
        for param in operation.get("parameters", []):
            schema = param.get("schema", {})
            param_info = {
                "name": param.get("name", ""),
                "type": schema.get("type", "string"),
                "location": param.get("in", "query"),
                "required": param.get("required", False),
                "description": param.get("description", ""),
                "default": schema.get("default"),
                "enum": schema.get("enum"),
                "example": param.get("example") or schema.get("example")
            }
            params.append(param_info)
        
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
                            "description": prop_schema.get("description", ""),
                            "default": prop_schema.get("default"),
                            "enum": prop_schema.get("enum"),
                            "example": prop_schema.get("example")
                        })
        
        return params
    
    # ---------------------------------------------------------------- held writes (MCP)

    def _hold_write(self, api_name: str, endpoint: "APIEndpoint", params: Dict[str, Any]) -> Dict[str, Any]:
        for key in [k for k, w in self.held_writes.items() if time.time() - w["created"] > 6 * HOLD_TTL_S]:
            self.held_writes.pop(key, None)  # ponytail: prune on write; a few dozen a day at most
        write_id = secrets.token_urlsafe(18)
        self.held_writes[write_id] = {"api": api_name, "endpoint": endpoint.name, "method": endpoint.method.upper(),
                                      "path": endpoint.path, "params": params, "created": time.time(),
                                      "status": "pending", "result": None}
        return {"success": False, "status": "pending_approval", "approval_id": write_id,
                "approval_url": APPROVE_URL + write_id,
                "message": (f"This {endpoint.method.upper()} changes shared demo data, so it was NOT sent. Ask the "
                            f"user to open approval_url and approve it within {HOLD_TTL_S // 60} minutes, then call "
                            f"call_api again with the same api_name and endpoint_name and parameters "
                            f'{{"approval_id": "{write_id}"}} to get the result.')}

    def _held_outcome(self, write_id: str) -> Dict[str, Any]:
        write = self.held_writes.get(write_id)
        if write is None:
            return {"success": False, "status": "unknown", "message": "No held write with this approval_id."}
        self._expire(write)
        if write["status"] == "approved":
            return {"success": True, "status": "approved", "data": write["result"]}
        messages = {"pending": "Not approved yet; ask the user to open the approval link.",
                    "rejected": "The user rejected it; it was not sent.",
                    "expired": "Nobody approved it in time; it was not sent.",
                    "failed": f"Approved, but the bank call failed: {write.get('error')}"}
        return {"success": False, "status": write["status"], "message": messages.get(write["status"], "")}

    def _expire(self, write: Dict[str, Any]) -> None:
        if write["status"] == "pending" and time.time() - write["created"] > HOLD_TTL_S:
            write["status"] = "expired"

    def held_write(self, write_id: str) -> Optional[Dict[str, Any]]:
        """What the approval page shows; None when there is no such write."""
        write = self.held_writes.get(write_id)
        if write is None:
            return None
        self._expire(write)
        left = max(0, int(HOLD_TTL_S - (time.time() - write["created"])))
        return {k: write[k] for k in ("api", "endpoint", "method", "path", "params", "status")} | {"seconds_left": left}

    async def decide_write(self, write_id: str, approve: bool) -> Optional[Dict[str, Any]]:
        """Approve (send it now) or reject a pending write; None when it is not pending."""
        write = self.held_writes.get(write_id)
        if write is None:
            return None
        self._expire(write)
        if write["status"] != "pending":
            return None
        if not approve:
            write["status"] = "rejected"
            return self.held_write(write_id)
        write["status"] = "sending"
        try:
            write["result"] = await self._execute_api_call(api_name=write["api"], endpoint_name=write["endpoint"],
                                                           params=dict(write["params"]))
            write["status"] = "approved"
        except Exception as e:  # the outcome is reported to the MCP client, not raised at the approver
            write["status"], write["error"] = "failed", f"{type(e).__name__}: {e}"[:300]
        return self.held_write(write_id)

    def run(self, **kwargs):
        """Run the MCP server."""
        if self.mcp:
            return self.mcp.run(**kwargs)
        else:
            logger.warning("Cannot run MCP server - FastMCP not available")
            return None

# Maintain backward compatibility
APIWeaver = BaNCSAPIWeaver