# backend/knowledge_base.py
"""
Unified API Knowledge Base for TCS BAnCS
Fixed version with proper async method
"""

import json
import yaml
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import httpx
from fuzzywuzzy import fuzz, process
import logging
import re

from .config import settings

# Try importing MCP components
try:
    from mcp import Tool, Resource
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Tool = None
    Resource = None

logger = logging.getLogger(__name__)

class UnifiedAPIKnowledgeBase:
    """
    Unified knowledge base that automatically generates MCP tools from API specifications
    """
    
    def __init__(self):
        self.endpoints = {}
        self.mcp_tools = []
        self.api_categories = {}
        self.http_client = httpx.AsyncClient(
            timeout=settings.api_timeout,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        
        # Initialize with default endpoints
        self._initialize_default_endpoints()
        
        # Load API specifications synchronously during init
        self._load_api_specifications()
        
        # Automatically generate MCP tools from loaded specifications
        if settings.mcp_enabled and MCP_AVAILABLE:
            self._auto_generate_mcp_tools()
    
    def _initialize_default_endpoints(self):
        """Initialize with default BAnCS endpoints"""
        self.endpoints = {
            "GetAccountBalance": {
                "method": "GET",
                "url": "/Core/accountManagement/account/balanceDetails/{accountReference}",
                "description": "Retrieve account balance details",
                "category": "account",
                "parameters": {
                    "accountReference": {
                        "type": "string",
                        "required": True,
                        "description": "15-digit account reference number",
                        "example": "101000000101814"
                    }
                },
                "mcp_enabled": True,
                "mcp_name": "get_account_balance"
            },
            "GetAccountDetails": {
                "method": "GET",
                "url": "/Core/accountManagement/account/details/{accountReference}",
                "description": "Retrieve detailed account information",
                "category": "account",
                "parameters": {
                    "accountReference": {
                        "type": "string",
                        "required": True,
                        "description": "15-digit account reference number"
                    }
                },
                "mcp_enabled": True,
                "mcp_name": "get_account_details"
            },
            "GetCustomerDetails": {
                "method": "GET",
                "url": "/Core/customerManagement/customer/viewDetails",
                "description": "Retrieve customer details",
                "category": "customer",
                "parameters": {
                    "CustomerID": {
                        "type": "string",
                        "required": True,
                        "description": "Customer ID",
                        "example": "146025"
                    }
                },
                "mcp_enabled": True,
                "mcp_name": "get_customer_details"
            },
            "CreateCustomer": {
                "method": "POST",
                "url": "/Core/customerManagement/customer",
                "description": "Create a new customer",
                "category": "customer",
                "parameters": {
                    "customerType": {
                        "type": "integer",
                        "required": True,
                        "description": "Type of customer (1=Individual, 2=Corporate)"
                    },
                    "customerFullName": {
                        "type": "string",
                        "required": True,
                        "description": "Full name of the customer"
                    },
                    "nationality": {
                        "type": "string",
                        "required": True,
                        "description": "Customer nationality code"
                    }
                },
                "mcp_enabled": True,
                "mcp_name": "create_customer"
            },
            "GetLoanDetails": {
                "method": "GET",
                "url": "/Core/loanManagement/loan/details/{loanReference}",
                "description": "Retrieve loan details",
                "category": "loan",
                "parameters": {
                    "loanReference": {
                        "type": "string",
                        "required": True,
                        "description": "Loan reference number"
                    }
                },
                "mcp_enabled": True,
                "mcp_name": "get_loan_details"
            },
            "CreateTransaction": {
                "method": "POST",
                "url": "/Core/bookingManagement/booking",
                "description": "Create a new transaction/booking",
                "category": "transaction",
                "parameters": {
                    "from_account": {
                        "type": "string",
                        "required": True,
                        "description": "Source account number"
                    },
                    "to_account": {
                        "type": "string",
                        "required": True,
                        "description": "Destination account number"
                    },
                    "amount": {
                        "type": "number",
                        "required": True,
                        "description": "Transaction amount"
                    },
                    "currency": {
                        "type": "string",
                        "required": True,
                        "description": "Currency code (e.g., USD, EUR)"
                    }
                },
                "mcp_enabled": True,
                "mcp_name": "create_transaction"
            }
        }
    
    def _load_api_specifications(self):
        """Load API specifications from various sources"""
        sources_loaded = []
        
        # Try loading from different sources
        spec_dir = Path("api_specs")
        if spec_dir.exists():
            # Load OpenAPI specs
            for file_path in spec_dir.glob("*.yaml"):
                if self._load_openapi_spec(file_path):
                    sources_loaded.append(f"OpenAPI: {file_path.name}")
            
            for file_path in spec_dir.glob("*.yml"):
                if self._load_openapi_spec(file_path):
                    sources_loaded.append(f"OpenAPI: {file_path.name}")
            
            # Load JSON specs (could be OpenAPI or Postman)
            for file_path in spec_dir.glob("*.json"):
                with open(file_path) as f:
                    try:
                        data = json.load(f)
                        if "info" in data and "_postman_id" in data.get("info", {}):
                            if self._load_postman_collection(file_path):
                                sources_loaded.append(f"Postman: {file_path.name}")
                        else:
                            if self._load_openapi_spec(file_path):
                                sources_loaded.append(f"OpenAPI: {file_path.name}")
                    except:
                        pass
        
        # Try loading from collections directory
        collections_dir = Path("collections")
        if collections_dir.exists():
            for file_path in collections_dir.glob("*.json"):
                if self._load_postman_collection(file_path):
                    sources_loaded.append(f"Postman: {file_path.name}")
        
        logger.info(f"Loaded API specifications from: {', '.join(sources_loaded) if sources_loaded else 'None (using defaults)'}")
        logger.info(f"Total endpoints loaded: {len(self.endpoints)}")
    
    def _load_openapi_spec(self, file_path: Path) -> bool:
        """Load OpenAPI specification"""
        try:
            with open(file_path) as f:
                if file_path.suffix in ['.yaml', '.yml']:
                    spec = yaml.safe_load(f)
                else:
                    spec = json.load(f)
            
            # Parse OpenAPI spec and add endpoints
            if 'paths' in spec:
                for path, methods in spec['paths'].items():
                    for method, details in methods.items():
                        if method.lower() in ['get', 'post', 'put', 'delete', 'patch']:
                            endpoint_name = details.get('operationId', f"{method}_{path}").replace('/', '_')
                            self.endpoints[endpoint_name] = {
                                "method": method.upper(),
                                "url": path,
                                "description": details.get('summary', details.get('description', '')),
                                "category": details.get('tags', ['general'])[0] if details.get('tags') else 'general',
                                "parameters": self._parse_openapi_parameters(details),
                                "mcp_enabled": True,
                                "mcp_name": endpoint_name.lower()
                            }
            
            logger.info(f"Loaded OpenAPI spec from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading OpenAPI spec from {file_path}: {e}")
            return False
    
    def _load_postman_collection(self, file_path: Path) -> bool:
        """Load Postman collection"""
        try:
            with open(file_path) as f:
                collection = json.load(f)
            
            # Parse Postman collection
            if 'item' in collection:
                self._parse_postman_items(collection['item'])
            
            logger.info(f"Loaded Postman collection from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading Postman collection from {file_path}: {e}")
            return False
    
    def _parse_postman_items(self, items, category='general'):
        """Recursively parse Postman collection items"""
        for item in items:
            if 'item' in item:
                # It's a folder, recurse
                self._parse_postman_items(item['item'], item.get('name', category))
            elif 'request' in item:
                # It's a request
                request = item['request']
                endpoint_name = item.get('name', 'Unknown').replace(' ', '_')
                
                self.endpoints[endpoint_name] = {
                    "method": request.get('method', 'GET'),
                    "url": self._extract_postman_url(request),
                    "description": item.get('description', ''),
                    "category": category,
                    "parameters": self._extract_postman_parameters(request),
                    "mcp_enabled": True,
                    "mcp_name": endpoint_name.lower()
                }
    
    def _extract_postman_url(self, request) -> str:
        """Extract URL from Postman request"""
        url = request.get('url', {})
        if isinstance(url, str):
            return url
        elif isinstance(url, dict):
            path = url.get('path', [])
            if isinstance(path, list):
                return '/' + '/'.join(path)
            return url.get('raw', '')
        return ''
    
    def _extract_postman_parameters(self, request) -> Dict[str, Any]:
        """Extract parameters from Postman request"""
        params = {}
        
        # Query parameters
        url = request.get('url', {})
        if isinstance(url, dict) and 'query' in url:
            for param in url['query']:
                params[param.get('key')] = {
                    "type": "string",
                    "required": not param.get('disabled', False),
                    "description": param.get('description', '')
                }
        
        # Path variables
        if isinstance(url, dict) and 'variable' in url:
            for var in url['variable']:
                params[var.get('key')] = {
                    "type": "string",
                    "required": True,
                    "description": var.get('description', '')
                }
        
        return params
    
    def _parse_openapi_parameters(self, operation) -> Dict[str, Any]:
        """Parse parameters from OpenAPI operation"""
        params = {}
        
        if 'parameters' in operation:
            for param in operation['parameters']:
                params[param.get('name')] = {
                    "type": param.get('schema', {}).get('type', 'string'),
                    "required": param.get('required', False),
                    "description": param.get('description', ''),
                    "in": param.get('in', 'query')
                }
        
        return params
    
    def _auto_generate_mcp_tools(self):
        """Automatically generate MCP tools from loaded specifications"""
        logger.info("Auto-generating MCP tools from API endpoints...")
        
        if not MCP_AVAILABLE:
            logger.warning("MCP not available, skipping tool generation")
            return
        
        for endpoint_name, endpoint in self.endpoints.items():
            if endpoint.get("mcp_enabled"):
                tool = self._create_mcp_tool(endpoint_name, endpoint)
                if tool:
                    self.mcp_tools.append(tool)
        
        logger.info(f"Generated {len(self.mcp_tools)} MCP tools")
    
    def _create_mcp_tool(self, name: str, endpoint: Dict[str, Any]):
        """Create an MCP tool from an endpoint"""
        if not Tool:
            return None
        
        return Tool(
            name=endpoint.get("mcp_name", name.lower()),
            description=endpoint["description"],
            inputSchema={
                "type": "object",
                "properties": endpoint.get("parameters", {}),
                "required": [
                    k for k, v in endpoint.get("parameters", {}).items()
                    if v.get("required", False)
                ]
            }
        )
    
    async def load_specifications(self):
        """Async method to load specifications - for compatibility with main.py"""
        # Since we already load in __init__, this is just a placeholder
        # You could move the loading logic here if you want it to be async
        logger.info(f"Specifications already loaded: {len(self.endpoints)} endpoints")
        return self.endpoints
    
    def search_endpoints(self, query: str) -> List[Tuple[str, int]]:
        """Search for relevant endpoints"""
        if not query:
            return []
        
        results = process.extract(
            query,
            self.endpoints.keys(),
            scorer=fuzz.token_sort_ratio,
            limit=5
        )
        
        return [(name, score) for name, score in results if score > settings.search_threshold]
    
    def get_endpoint_by_mcp_name(self, mcp_name: str) -> Optional[Dict[str, Any]]:
        """Get endpoint by MCP tool name"""
        for endpoint in self.endpoints.values():
            if endpoint.get("mcp_name") == mcp_name:
                return endpoint
        return None

# ============= ADD THIS TO THE END OF backend/knowledge_base.py =============

    async def list_endpoints(self, category=None):
        """List all endpoints"""
        endpoints_list = []
        for name, endpoint in self.endpoints.items():
            endpoint_info = {
                "name": name,
                "method": endpoint.get("method", "GET"),
                "path": endpoint.get("url", "/"),
                "description": endpoint.get("description", ""),
                "category": endpoint.get("category", "general")
            }
            
            # Filter by category if specified
            if category is None or endpoint_info["category"] == category:
                endpoints_list.append(endpoint_info)
        
        return endpoints_list
    
    async def get_cookbook_recipes(self, category=None):
        """Get cookbook recipes from endpoints"""
        recipes = []
        
        # Group endpoints by category
        categories = {}
        for name, endpoint in self.endpoints.items():
            cat = endpoint.get("category", "general")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((name, endpoint))
        
        # Create recipes from endpoints
        recipe_id = 0
        for cat_name, endpoints in categories.items():
            if category and category != cat_name:
                continue
                
            for endpoint_name, endpoint in endpoints:
                recipe = {
                    "id": f"recipe_{recipe_id}",
                    "title": endpoint_name,
                    "description": endpoint.get("description", f"API endpoint: {endpoint_name}"),
                    "category": cat_name,
                    "icon": self._get_icon_for_category(cat_name),
                    "tags": [
                        endpoint.get("method", "GET"),
                        cat_name
                    ],
                    "endpoint": endpoint_name,
                    "method": endpoint.get("method", "GET"),
                    "url": endpoint.get("url", "/"),
                    "parameters": endpoint.get("parameters", {})
                }
                recipes.append(recipe)
                recipe_id += 1
        
        return recipes
    
    async def get_recipe(self, recipe_id):
        """Get a specific recipe"""
        recipes = await self.get_cookbook_recipes()
        for recipe in recipes:
            if recipe["id"] == recipe_id:
                return recipe
        return None
    
    async def generate_openapi_spec(self, version="3.0"):
        """Generate OpenAPI specification from endpoints"""
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
                    "description": "BAnCS Demo Server"
                }
            ],
            "paths": {},
            "components": {
                "securitySchemes": {
                    "CustomHeaders": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "entity"
                    }
                }
            }
        }
        
        # Add paths from endpoints
        for name, endpoint in self.endpoints.items():
            path = endpoint.get("url", "/")
            method = endpoint.get("method", "GET").lower()
            
            if path not in spec["paths"]:
                spec["paths"][path] = {}
            
            operation = {
                "summary": endpoint.get("description", name),
                "operationId": name,
                "tags": [endpoint.get("category", "general")],
                "responses": {
                    "200": {
                        "description": "Successful response",
                        "content": {
                            "application/json": {
                                "schema": {"type": "object"}
                            }
                        }
                    },
                    "400": {"description": "Bad request"},
                    "401": {"description": "Unauthorized"},
                    "500": {"description": "Internal server error"}
                }
            }
            
            # Add parameters
            if endpoint.get("parameters"):
                operation["parameters"] = []
                for param_name, param_info in endpoint["parameters"].items():
                    param_spec = {
                        "name": param_name,
                        "in": "path" if "{" + param_name + "}" in path else "query",
                        "required": param_info.get("required", False),
                        "schema": {
                            "type": param_info.get("type", "string")
                        },
                        "description": param_info.get("description", "")
                    }
                    operation["parameters"].append(param_spec)
            
            spec["paths"][path][method] = operation
        
        return spec
    
    def _get_icon_for_category(self, category):
        """Get an icon for a category"""
        icons = {
            "account": "💰",
            "customer": "👤",
            "loan": "💳",
            "transaction": "💸",
            "general": "📚",
            "authentication": "🔐",
            "reporting": "📊"
        }
        return icons.get(category, "📌")
    
    def export_mcp_tools(self):
        """Export MCP tools information"""
        tools = []
        for tool in self.mcp_tools:
            tools.append({
                'name': tool.name,
                'description': tool.description,
                'inputSchema': tool.inputSchema
            })
        return tools