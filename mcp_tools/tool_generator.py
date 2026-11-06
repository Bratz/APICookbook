# mcp/tool_generator.py (NEW - Optional utility for documentation)
"""
Dynamic tool documentation generator
Generates examples and documentation from discovered tools
"""

import json
from typing import Dict, List, Any
from datetime import datetime

class MCPToolDocGenerator:
    """Generates documentation and examples for auto-discovered MCP tools"""
    
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
    
    def generate_tool_documentation(self) -> str:
        """Generate markdown documentation for all tools"""
        doc = f"# MCP Tools Documentation\n\n"
        doc += f"*Generated: {datetime.now().isoformat()}*\n\n"
        doc += f"Total Tools Available: {len(self.kb.mcp_tools)}\n\n"
        
        # Group tools by category
        tools_by_category = {}
        for tool in self.kb.mcp_tools:
            # Find the endpoint for this tool
            endpoint = self.kb.get_endpoint_by_mcp_name(tool.name)
            if endpoint:
                category = endpoint.get("category", "general")
                if category not in tools_by_category:
                    tools_by_category[category] = []
                tools_by_category[category].append((tool, endpoint))
        
        # Generate documentation for each category
        for category, tools in tools_by_category.items():
            doc += f"## {category.title()}\n\n"
            
            for tool, endpoint in tools:
                doc += f"### `{tool.name}`\n\n"
                doc += f"**Description**: {tool.description}\n\n"
                doc += f"**Endpoint**: `{endpoint['method']} {endpoint['url']}`\n\n"
                
                # Parameters
                if tool.inputSchema.get("properties"):
                    doc += "**Parameters**:\n\n"
                    for param_name, param_schema in tool.inputSchema["properties"].items():
                        required = "Required" if param_name in tool.inputSchema.get("required", []) else "Optional"
                        doc += f"- `{param_name}` ({param_schema.get('type', 'string')}, {required}): {param_schema.get('description', '')}\n"
                    doc += "\n"
                
                # Example usage
                doc += "**Example Usage**:\n\n```python\n"
                doc += self._generate_example_code(tool.name, tool.inputSchema)
                doc += "\n```\n\n---\n\n"
        
        return doc
    
    def generate_client_examples(self) -> str:
        """Generate example code for using the tools"""
        code = '''"""
Auto-generated MCP client examples
Generated from discovered API endpoints
"""

import asyncio
from backend.mcp_client import BaNCSMCPClient

async def example_usage():
    """Example usage of auto-discovered MCP tools"""
    
    async with BaNCSMCPClient() as client:
        # List all available tools
        tools = await client.list_available_tools()
        print(f"Found {len(tools)} MCP tools:")
        for tool in tools[:5]:  # Show first 5
            print(f"  - {tool['name']}: {tool['description']}")
        
'''
        
        # Generate examples for a few tools
        for i, tool in enumerate(self.kb.mcp_tools[:3]):
            code += f"""
        # Example {i+1}: {tool.name}
        print("\\nCalling {tool.name}...")
        result = await client.call_tool(
            "{tool.name}",
            {self._generate_example_params(tool.inputSchema)}
        )
        print(f"Result: {{result}}")
"""
        
        code += '''

if __name__ == "__main__":
    asyncio.run(example_usage())
'''
        return code
    
    def _generate_example_code(self, tool_name: str, schema: Dict[str, Any]) -> str:
        """Generate example code for a tool"""
        params = self._generate_example_params(schema)
        
        return f"""from mcp_client import BaNCSMCPClient

async with BaNCSMCPClient() as client:
    result = await client.call_tool(
        "{tool_name}",
        {json.dumps(params, indent=8)}
    )
    print(result)"""
    
    def _generate_example_params(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate example parameters based on schema"""
        params = {}
        
        for param_name, param_schema in schema.get("properties", {}).items():
            param_type = param_schema.get("type", "string")
            
            # Generate example value based on type
            if "example" in param_schema:
                params[param_name] = param_schema["example"]
            elif "default" in param_schema:
                params[param_name] = param_schema["default"]
            elif param_type == "string":
                if "pattern" in param_schema:
                    # Use a sample that matches the pattern
                    if "account" in param_name.lower():
                        params[param_name] = "101000000101814"
                    elif "date" in param_name.lower():
                        params[param_name] = "20240101"
                    else:
                        params[param_name] = "example_value"
                else:
                    params[param_name] = f"example_{param_name}"
            elif param_type == "integer":
                params[param_name] = 1
            elif param_type == "number":
                params[param_name] = 100.00
            elif param_type == "boolean":
                params[param_name] = True
            elif param_type == "array":
                params[param_name] = []
            elif param_type == "object":
                params[param_name] = {}
        
        return params
    
    def export_tools_manifest(self) -> Dict[str, Any]:
        """Export tools manifest for external consumption"""
        manifest = {
            "version": "2.0.0",
            "generated": datetime.now().isoformat(),
            "server": {
                "name": "tcs-bancs-mcp",
                "description": "TCS BaNCS API MCP Server",
                "version": "2.0.0"
            },
            "tools": []
        }
        
        for tool in self.kb.mcp_tools:
            endpoint = self.kb.get_endpoint_by_mcp_name(tool.name)
            tool_info = {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema,
                "endpoint": {
                    "method": endpoint.get("method"),
                    "path": endpoint.get("url")
                } if endpoint else None
            }
            manifest["tools"].append(tool_info)
        
        return manifest
