"""
Dynamic API Specification Loader
Watches and automatically reloads API specifications
Following APIWeaver pattern for automatic tool generation
"""

import asyncio
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import hashlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging

from .config import settings

logger = logging.getLogger(__name__)

class SpecificationWatcher(FileSystemEventHandler):
    """Watches for changes in API specification files"""
    
    def __init__(self, callback):
        self.callback = callback
        self.last_modified = {}
        
    def on_modified(self, event):
        if not event.is_directory:
            file_path = Path(event.src_path)
            if self._is_spec_file(file_path):
                # Check if file actually changed (avoid duplicate events)
                file_hash = self._get_file_hash(file_path)
                if file_hash != self.last_modified.get(str(file_path)):
                    self.last_modified[str(file_path)] = file_hash
                    logger.info(f"API specification changed: {file_path}")
                    asyncio.create_task(self.callback(file_path))
    
    def _is_spec_file(self, file_path: Path) -> bool:
        """Check if file is an API specification"""
        spec_extensions = ['.yaml', '.yml', '.json']
        spec_patterns = ['openapi', 'swagger', 'postman', 'api_spec']
        
        return (
            file_path.suffix in spec_extensions and
            any(pattern in file_path.name.lower() for pattern in spec_patterns)
        )
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""

class APISpecificationLoader:
    """
    Loads and manages API specifications from multiple sources
    Automatically generates MCP tools from specifications
    """
    
    def __init__(self):
        self.specifications = {}
        self.spec_sources = []
        self.watcher = None
        self.observer = None
        self.endpoints_cache = {}
        self.tools_cache = {}
        self._init_spec_sources()
    
    def _init_spec_sources(self):
        """Initialize specification sources"""
        self.spec_sources = [
            # OpenAPI/Swagger specifications
            SpecSource(
                type="openapi",
                paths=[
                    "api_specs/openapi.yaml",
                    "api_specs/openapi.yml",
                    "api_specs/swagger.json",
                    "api_specs/openapi.json"
                ],
                parser=self.parse_openapi
            ),
            # Postman collections
            SpecSource(
                type="postman",
                paths=[
                    "api_specs/postman_collection.json",
                    "collections/*.json"
                ],
                parser=self.parse_postman
            ),
            # Custom JSON specifications
            SpecSource(
                type="custom",
                paths=[
                    "api_specs/api_spec.json",
                    "api_specs/endpoints.json"
                ],
                parser=self.parse_custom_json
            ),
            # AsyncAPI specifications
            SpecSource(
                type="asyncapi",
                paths=[
                    "api_specs/asyncapi.yaml",
                    "api_specs/asyncapi.json"
                ],
                parser=self.parse_asyncapi
            )
        ]
    
    async def load_all_specifications(self) -> Dict[str, Any]:
        """Load all API specifications from configured sources"""
        all_endpoints = {}
        
        for source in self.spec_sources:
            endpoints = await self._load_from_source(source)
            all_endpoints.update(endpoints)
        
        # Cache the loaded endpoints
        self.endpoints_cache = all_endpoints
        
        # Generate MCP tools from endpoints
        if settings.mcp_enabled:
            self.tools_cache = self._generate_tools_from_endpoints(all_endpoints)
        
        logger.info(f"Loaded {len(all_endpoints)} endpoints from specifications")
        return all_endpoints
    
    async def _load_from_source(self, source: 'SpecSource') -> Dict[str, Any]:
        """Load specifications from a source"""
        endpoints = {}
        
        for path_pattern in source.paths:
            for file_path in Path(".").glob(path_pattern):
                if file_path.exists():
                    try:
                        spec = await self._load_spec_file(file_path)
                        parsed_endpoints = source.parser(spec, file_path)
                        endpoints.update(parsed_endpoints)
                        logger.info(f"Loaded {len(parsed_endpoints)} endpoints from {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to load {file_path}: {e}")
        
        return endpoints
    
    async def _load_spec_file(self, file_path: Path) -> Dict[str, Any]:
        """Load a specification file"""
        with open(file_path, 'r') as f:
            if file_path.suffix in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            else:
                return json.load(f)
    
    def parse_openapi(self, spec: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
        """Parse OpenAPI/Swagger specification"""
        endpoints = {}
        
        # Determine OpenAPI version
        is_openapi_3 = "openapi" in spec and spec["openapi"].startswith("3")
        
        # Get base configuration
        base_url = self._get_base_url(spec, is_openapi_3)
        global_security = spec.get("security", [])
        
        # Parse paths
        for path, path_item in spec.get("paths", {}).items():
            # Handle path-level parameters
            path_params = path_item.get("parameters", [])
            
            for method, operation in path_item.items():
                if method.upper() not in ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]:
                    continue
                
                endpoint_id = operation.get("operationId", f"{method}_{path}".replace("/", "_"))
                
                # Merge parameters
                parameters = self._merge_parameters(
                    path_params,
                    operation.get("parameters", [])
                )
                
                # Parse request body (OpenAPI 3.0)
                if is_openapi_3 and "requestBody" in operation:
                    body_params = self._parse_request_body(operation["requestBody"])
                    parameters.extend(body_params)
                
                # Create endpoint definition
                endpoint = {
                    "id": endpoint_id,
                    "name": operation.get("summary", endpoint_id),
                    "description": operation.get("description", ""),
                    "method": method.upper(),
                    "path": path,
                    "base_url": base_url,
                    "parameters": parameters,
                    "responses": operation.get("responses", {}),
                    "security": operation.get("security", global_security),
                    "tags": operation.get("tags", []),
                    "deprecated": operation.get("deprecated", False),
                    "x-mcp-enabled": operation.get("x-mcp-enabled", True),
                    "x-mcp-tool-name": operation.get("x-mcp-tool-name"),
                    "source": str(file_path),
                    "source_type": "openapi"
                }
                
                endpoints[endpoint_id] = endpoint
        
        return endpoints
    
    def parse_postman(self, collection: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
        """Parse Postman collection"""
        endpoints = {}
        
        def parse_items(items: List[Dict], folder_path: str = ""):
            for item in items:
                if "item" in item:
                    # Folder - recurse
                    new_path = f"{folder_path}/{item.get('name', '')}" if folder_path else item.get('name', '')
                    parse_items(item["item"], new_path)
                    
                elif "request" in item:
                    # Request item
                    request = item["request"]
                    endpoint_id = item.get("name", "").replace(" ", "_")
                    
                    # Parse URL
                    url_data = request.get("url", {})
                    if isinstance(url_data, str):
                        path = self._extract_path_from_url(url_data)
                        base_url = url_data.split(path)[0] if path in url_data else ""
                    else:
                        path = "/" + "/".join(url_data.get("path", []))
                        protocol = url_data.get("protocol", "https")
                        host = ".".join(url_data.get("host", []))
                        base_url = f"{protocol}://{host}"
                    
                    # Parse parameters
                    parameters = []
                    
                    # Query parameters
                    if isinstance(url_data, dict):
                        for param in url_data.get("query", []):
                            parameters.append({
                                "name": param.get("key"),
                                "in": "query",
                                "description": param.get("description", ""),
                                "required": not param.get("disabled", False),
                                "type": "string",
                                "default": param.get("value")
                            })
                    
                    # Headers
                    for header in request.get("header", []):
                        if header.get("key") not in ["Content-Type", "Accept"]:
                            parameters.append({
                                "name": header.get("key"),
                                "in": "header",
                                "description": header.get("description", ""),
                                "required": not header.get("disabled", False),
                                "type": "string"
                            })
                    
                    # Body
                    if request.get("body"):
                        body_params = self._parse_postman_body(request["body"])
                        parameters.extend(body_params)
                    
                    # Create endpoint
                    endpoint = {
                        "id": endpoint_id,
                        "name": item.get("name", endpoint_id),
                        "description": item.get("description", ""),
                        "method": request.get("method", "GET"),
                        "path": path,
                        "base_url": base_url,
                        "parameters": parameters,
                        "tags": [folder_path] if folder_path else [],
                        "x-mcp-enabled": True,
                        "source": str(file_path),
                        "source_type": "postman"
                    }
                    
                    endpoints[endpoint_id] = endpoint
        
        # Parse collection
        parse_items(collection.get("item", []))
        return endpoints
    
    def parse_custom_json(self, spec: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
        """Parse custom JSON specification"""
        endpoints = {}
        
        for endpoint_data in spec.get("endpoints", []):
            endpoint_id = endpoint_data.get("id", endpoint_data.get("name", "").replace(" ", "_"))
            
            # Ensure required fields
            endpoint = {
                "id": endpoint_id,
                "source": str(file_path),
                "source_type": "custom",
                "x-mcp-enabled": True
            }
            
            # Copy all fields from custom spec
            endpoint.update(endpoint_data)
            
            endpoints[endpoint_id] = endpoint
        
        return endpoints
    
    def parse_asyncapi(self, spec: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
        """Parse AsyncAPI specification for event-driven APIs"""
        endpoints = {}
        
        # Parse channels (similar to paths in OpenAPI)
        for channel_name, channel in spec.get("channels", {}).items():
            for operation_type in ["publish", "subscribe"]:
                if operation_type in channel:
                    operation = channel[operation_type]
                    endpoint_id = operation.get("operationId", f"{operation_type}_{channel_name}")
                    
                    endpoint = {
                        "id": endpoint_id,
                        "name": operation.get("summary", endpoint_id),
                        "description": operation.get("description", ""),
                        "method": operation_type.upper(),
                        "path": channel_name,
                        "parameters": self._parse_asyncapi_parameters(operation),
                        "tags": operation.get("tags", []),
                        "x-mcp-enabled": True,
                        "source": str(file_path),
                        "source_type": "asyncapi"
                    }
                    
                    endpoints[endpoint_id] = endpoint
        
        return endpoints
    
    def _get_base_url(self, spec: Dict[str, Any], is_openapi_3: bool) -> str:
        """Extract base URL from specification"""
        if is_openapi_3:
            servers = spec.get("servers", [])
            if servers:
                return servers[0].get("url", "")
        else:
            # Swagger 2.0
            scheme = "https" if "https" in spec.get("schemes", ["https"]) else "http"
            host = spec.get("host", "localhost")
            base_path = spec.get("basePath", "")
            return f"{scheme}://{host}{base_path}"
        
        return ""
    
    def _merge_parameters(self, *param_lists) -> List[Dict[str, Any]]:
        """Merge multiple parameter lists, removing duplicates"""
        merged = {}
        for param_list in param_lists:
            for param in param_list:
                param_key = f"{param.get('name')}_{param.get('in')}"
                merged[param_key] = param
        return list(merged.values())
    
    def _parse_request_body(self, request_body: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse OpenAPI 3.0 request body"""
        parameters = []
        
        content = request_body.get("content", {})
        for media_type, media_type_obj in content.items():
            if media_type == "application/json":
                schema = media_type_obj.get("schema", {})
                parameters.extend(self._schema_to_parameters(schema))
        
        return parameters
    
    def _parse_postman_body(self, body: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse Postman request body"""
        parameters = []
        
        mode = body.get("mode")
        if mode == "raw":
            try:
                raw_body = json.loads(body.get("raw", "{}"))
                parameters = self._json_to_parameters(raw_body)
            except:
                pass
        elif mode == "formdata" or mode == "urlencoded":
            for param in body.get(mode, []):
                parameters.append({
                    "name": param.get("key"),
                    "in": "body",
                    "description": param.get("description", ""),
                    "required": not param.get("disabled", False),
                    "type": param.get("type", "string")
                })
        
        return parameters
    
    def _parse_asyncapi_parameters(self, operation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse AsyncAPI operation parameters"""
        parameters = []
        
        # Parse message schema
        message = operation.get("message", {})
        if "payload" in message:
            parameters.extend(self._schema_to_parameters(message["payload"]))
        
        return parameters
    
    def _schema_to_parameters(self, schema: Dict[str, Any], prefix: str = "") -> List[Dict[str, Any]]:
        """Convert JSON schema to parameter list"""
        parameters = []
        
        if schema.get("type") == "object":
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            
            for prop_name, prop_schema in properties.items():
                param_name = f"{prefix}.{prop_name}" if prefix else prop_name
                
                if prop_schema.get("type") == "object":
                    # Recurse for nested objects
                    parameters.extend(self._schema_to_parameters(prop_schema, param_name))
                else:
                    parameters.append({
                        "name": param_name,
                        "in": "body",
                        "type": prop_schema.get("type", "string"),
                        "description": prop_schema.get("description", ""),
                        "required": prop_name in required,
                        "default": prop_schema.get("default"),
                        "enum": prop_schema.get("enum"),
                        "pattern": prop_schema.get("pattern"),
                        "minimum": prop_schema.get("minimum"),
                        "maximum": prop_schema.get("maximum")
                    })
        
        return parameters
    
    def _json_to_parameters(self, json_obj: Any, prefix: str = "") -> List[Dict[str, Any]]:
        """Convert JSON object to parameter list"""
        parameters = []
        
        if isinstance(json_obj, dict):
            for key, value in json_obj.items():
                param_name = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, dict):
                    parameters.extend(self._json_to_parameters(value, param_name))
                else:
                    parameters.append({
                        "name": param_name,
                        "in": "body",
                        "type": self._infer_type(value),
                        "description": "",
                        "required": False,
                        "example": value
                    })
        
        return parameters
    
    def _infer_type(self, value: Any) -> str:
        """Infer JSON schema type from value"""
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
    
    def _extract_path_from_url(self, url: str) -> str:
        """Extract path from URL"""
        if "://" in url:
            url = url.split("://", 1)[1]
        if "/" in url:
            return "/" + url.split("/", 1)[1].split("?")[0]
        return "/"
    
    def _generate_tools_from_endpoints(self, endpoints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate MCP tool definitions from endpoints"""
        tools = []
        
        for endpoint_id, endpoint in endpoints.items():
            if endpoint.get("x-mcp-enabled", True) and not endpoint.get("deprecated", False):
                tool = {
                    "name": endpoint.get("x-mcp-tool-name") or self._generate_tool_name(endpoint),
                    "description": endpoint.get("description") or endpoint.get("name", ""),
                    "endpoint_id": endpoint_id,
                    "inputSchema": self._generate_tool_schema(endpoint)
                }
                tools.append(tool)
        
        return tools
    
    def _generate_tool_name(self, endpoint: Dict[str, Any]) -> str:
        """Generate MCP tool name from endpoint"""
        # Use operationId if available
        if "id" in endpoint:
            name = endpoint["id"]
        else:
            # Generate from method and path
            method = endpoint.get("method", "").lower()
            path = endpoint.get("path", "").replace("/", "_").replace("{", "").replace("}", "")
            name = f"{method}{path}"
        
        # Convert to snake_case
        name = re.sub(r'[^\w]', '_', name)
        name = re.sub(r'_+', '_', name)
        name = name.lower().strip('_')
        
        return name
    
    def _generate_tool_schema(self, endpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Generate JSON schema for MCP tool from endpoint parameters"""
        properties = {}
        required = []
        
        for param in endpoint.get("parameters", []):
            if param.get("in") != "header":  # Skip headers for MCP tools
                param_name = param.get("name", "")
                param_schema = {
                    "type": param.get("type", "string"),
                    "description": param.get("description", "")
                }
                
                # Add constraints
                for field in ["enum", "pattern", "minimum", "maximum", "minLength", "maxLength"]:
                    if field in param:
                        param_schema[field] = param[field]
                
                if "default" in param:
                    param_schema["default"] = param["default"]
                
                properties[param_name] = param_schema
                
                if param.get("required", False):
                    required.append(param_name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }
    
    def start_watching(self, callback):
        """Start watching for specification file changes"""
        if not settings.api_spec_auto_reload:
            return
        
        self.watcher = SpecificationWatcher(callback)
        self.observer = Observer()
        
        # Watch directories
        watch_dirs = set()
        for source in self.spec_sources:
            for path_pattern in source.paths:
                path = Path(path_pattern)
                watch_dirs.add(str(path.parent))
        
        for directory in watch_dirs:
            if Path(directory).exists():
                self.observer.schedule(self.watcher, directory, recursive=True)
                logger.info(f"Watching directory for spec changes: {directory}")
        
        self.observer.start()
    
    def stop_watching(self):
        """Stop watching for file changes"""
        if self.observer:
            self.observer.stop()
            self.observer.join()

class SpecSource:
    """Represents a source of API specifications"""
    
    def __init__(self, type: str, paths: List[str], parser):
        self.type = type
        self.paths = paths
        self.parser = parser

# Import required modules
import re