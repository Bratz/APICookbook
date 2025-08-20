"""
Cookbook engine for parsing Postman collection and providing documentation.
This is separate from MCP server and only provides cookbook functionality.
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import re


class CookbookEngine:
    """Parse Postman collection for cookbook documentation only."""
    
    def __init__(self, collection_path: str = "collections/AWSBancs_Collection.json"):
        self.collection_path = Path(collection_path)
        self.endpoints = {}
        self.categories = {}
        self._load_collection()
    
    def _load_collection(self):
        """Load and parse Postman collection."""
        if not self.collection_path.exists():
            return
        
        with open(self.collection_path, 'r') as f:
            collection = json.load(f)
        
        # Parse collection items
        self._parse_items(collection.get("item", []))
    
    def _parse_items(self, items: List[Dict], category: str = "general"):
        """Parse Postman collection items."""
        for item in items:
            if "item" in item:
                # This is a folder
                folder_name = item.get("name", category)
                self._parse_items(item["item"], folder_name)
            elif "request" in item:
                # This is an endpoint
                self._parse_endpoint(item, category)
    
    def _parse_endpoint(self, item: Dict, category: str):
        """Parse a single endpoint from collection."""
        request = item.get("request", {})
        name = item.get("name", "Unknown")
        
        # Parse URL
        url_info = request.get("url", {})
        if isinstance(url_info, str):
            path = url_info
            base_url = ""
        else:
            raw_url = url_info.get("raw", "")
            path = "/" + "/".join(url_info.get("path", []))
            protocol = url_info.get("protocol", "https")
            host = url_info.get("host", [])
            if isinstance(host, list):
                host = ".".join(host)
            base_url = f"{protocol}://{host}"
        
        # Parse parameters
        parameters = {}
        
        # Query parameters
        if isinstance(url_info, dict):
            for param in url_info.get("query", []):
                parameters[param.get("key")] = {
                    "type": "query",
                    "value": param.get("value"),
                    "description": param.get("description", "")
                }
        
        # Headers
        for header in request.get("header", []):
            if header.get("key") in ["accountReference", "customerReference"]:
                parameters[header.get("key")] = {
                    "type": "header",
                    "value": header.get("value"),
                    "description": header.get("description", "")
                }
        
        # Body
        if request.get("body"):
            body = request["body"]
            if body.get("mode") == "raw":
                try:
                    body_json = json.loads(body.get("raw", "{}"))
                    # Store body example
                    parameters["_body_example"] = body_json
                except:
                    pass
        
        # Store endpoint
        endpoint_data = {
            "name": name,
            "method": request.get("method", "GET"),
            "path": path,
            "base_url": base_url,
            "full_url": base_url + path,
            "category": category,
            "parameters": parameters,
            "description": item.get("description", "")
        }
        
        self.endpoints[name] = endpoint_data
        
        # Add to category
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(name)
    
    def get_recipe(self, endpoint_name: str) -> Dict[str, Any]:
        """Get cookbook recipe for an endpoint."""
        endpoint = self.endpoints.get(endpoint_name)
        if not endpoint:
            return None
        
        # Build recipe
        recipe = {
            "title": endpoint["name"],
            "method": endpoint["method"],
            "url": endpoint["full_url"],
            "path": endpoint["path"],
            "category": endpoint["category"],
            "description": endpoint["description"] or f"API endpoint: {endpoint['name']}",
            "parameters": [],
            "headers": {
                "entity": "GPRDTTSTOU",
                "userId": "1",
                "languageCode": "1"
            },
            "examples": {}
        }
        
        # Process parameters
        for param_name, param_info in endpoint["parameters"].items():
            if param_name != "_body_example":
                recipe["parameters"].append({
                    "name": param_name,
                    "type": param_info["type"],
                    "example": param_info["value"],
                    "description": param_info["description"]
                })
        
        # Add body example if exists
        if "_body_example" in endpoint["parameters"]:
            recipe["body_example"] = endpoint["parameters"]["_body_example"]
        
        # Generate code examples
        recipe["examples"]["curl"] = self._generate_curl_example(endpoint)
        recipe["examples"]["python"] = self._generate_python_example(endpoint)
        recipe["examples"]["javascript"] = self._generate_js_example(endpoint)
        
        return recipe
    
    def _generate_curl_example(self, endpoint: Dict) -> str:
        """Generate cURL example."""
        url = endpoint["full_url"]
        
        # Add query parameters
        query_params = []
        for name, info in endpoint["parameters"].items():
            if info.get("type") == "query" and info.get("value"):
                query_params.append(f"{name}={info['value']}")
        
        if query_params:
            url += "?" + "&".join(query_params)
        
        curl = f"""curl -X {endpoint['method']} \\
  "{url}" \\
  -H "entity: GPRDTTSTOU" \\
  -H "userId: 1" \\
  -H "languageCode: 1" """
        
        # Add custom headers
        for name, info in endpoint["parameters"].items():
            if info.get("type") == "header":
                curl += f"""\\
  -H "{name}: {info.get('value', '')}" """
        
        # Add body if exists
        if "_body_example" in endpoint["parameters"]:
            body = json.dumps(endpoint["parameters"]["_body_example"])
            curl += f"""\\
  -H "Content-Type: application/json" \\
  -d '{body}'"""
        
        return curl
    
    def _generate_python_example(self, endpoint: Dict) -> str:
        """Generate Python example."""
        code = f"""import requests

url = "{endpoint['full_url']}"
headers = {{
    "entity": "GPRDTTSTOU",
    "userId": "1",
    "languageCode": "1"
}}"""
        
        # Add custom headers
        for name, info in endpoint["parameters"].items():
            if info.get("type") == "header":
                code += f"""
headers["{name}"] = "{info.get('value', '')}" """
        
        # Add query params
        query_params = {}
        for name, info in endpoint["parameters"].items():
            if info.get("type") == "query" and info.get("value"):
                query_params[name] = info["value"]
        
        if query_params:
            code += f"""
params = {json.dumps(query_params, indent=4)}"""
        
        # Add body
        if "_body_example" in endpoint["parameters"]:
            code += f"""
data = {json.dumps(endpoint["parameters"]["_body_example"], indent=4)}

response = requests.{endpoint['method'].lower()}(url, headers=headers, json=data"""
            if query_params:
                code += ", params=params"
            code += ")"
        else:
            code += f"""

response = requests.{endpoint['method'].lower()}(url, headers=headers"""
            if query_params:
                code += ", params=params"
            code += ")"
        
        code += """
print(response.json())"""
        
        return code
    
    def _generate_js_example(self, endpoint: Dict) -> str:
        """Generate JavaScript example."""
        code = f"""const url = "{endpoint['full_url']}";
const headers = {{
    "entity": "GPRDTTSTOU",
    "userId": "1",
    "languageCode": "1"
}};"""
        
        # Add body if exists
        if "_body_example" in endpoint["parameters"]:
            code += f"""

const data = {json.dumps(endpoint["parameters"]["_body_example"], indent=4)};

fetch(url, {{
    method: "{endpoint['method']}",
    headers: headers,
    body: JSON.stringify(data)
}})"""
        else:
            code += f"""

fetch(url, {{
    method: "{endpoint['method']}",
    headers: headers
}})"""
        
        code += """
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error('Error:', error));"""
        
        return code
    
    def list_endpoints(self) -> List[Dict[str, Any]]:
        """List all endpoints for cookbook."""
        result = []
        for name, endpoint in self.endpoints.items():
            result.append({
                "name": name,
                "method": endpoint["method"],
                "path": endpoint["path"],
                "category": endpoint["category"],
                "description": endpoint["description"]
            })
        return result