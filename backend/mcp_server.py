"""
MCP Server Implementation for TCS BAnCS
Provides Model Context Protocol tools for AI integration
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

# Try importing MCP components
try:
    from mcp import Server, Tool, Resource
    from mcp.server import stdio
    from mcp.types import (
        TextContent,
        ImageContent,
        CallToolResult,
        ListToolsResult,
        ListResourcesResult,
        ReadResourceResult,
        GetPromptResult,
        ListPromptsResult,
        Prompt,
        PromptMessage
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Create dummy classes for when MCP is not available
    class Server:
        def __init__(self, name: str):
            self.name = name
    
    class Tool:
        def __init__(self, name: str, description: str, inputSchema: dict):
            self.name = name
            self.description = description
            self.inputSchema = inputSchema
    
    TextContent = dict
    CallToolResult = dict
    ListToolsResult = dict

from config import settings

logger = logging.getLogger(__name__)

class IntegratedMCPServer(Server):
    """MCP Server integrated with the TCS BAnCS API chatbot"""
    
    def __init__(self, knowledge_base):
        super().__init__(settings.mcp_server_name)
        self.kb = knowledge_base
        self.logger = logging.getLogger(__name__)
        self.execution_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "tool_usage": {}
        }
        
        # Register handlers
        if MCP_AVAILABLE:
            self._register_handlers()
    
    def _register_handlers(self):
        """Register MCP protocol handlers"""
        self.list_tools_handler = self.list_tools
        self.call_tool_handler = self.call_tool
        self.list_resources_handler = self.list_resources
        self.read_resource_handler = self.read_resource
        self.list_prompts_handler = self.list_prompts
        self.get_prompt_handler = self.get_prompt
    
    async def list_tools(self) -> ListToolsResult:
        """List all available MCP tools"""
        if not MCP_AVAILABLE:
            return {"tools": []}
        
        self.logger.info(f"Listing {len(self.kb.mcp_tools)} MCP tools")
        return ListToolsResult(tools=self.kb.mcp_tools)
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """Execute an MCP tool"""
        if not MCP_AVAILABLE:
            return {
                "content": [{"type": "text", "text": "MCP not available"}],
                "isError": True
            }
        
        self.logger.info(f"MCP tool call: {name} with args: {arguments}")
        self.execution_stats["total_calls"] += 1
        self.execution_stats["tool_usage"][name] = self.execution_stats["tool_usage"].get(name, 0) + 1
        
        try:
            # Find the corresponding endpoint
            endpoint = self.kb.get_endpoint_by_mcp_name(name)
            if not endpoint:
                raise ValueError(f"Unknown MCP tool: {name}")
            
            # Convert MCP arguments to API parameters
            api_params = self._convert_mcp_to_api_params(arguments, endpoint)
            
            # Execute the API call
            result = await self.kb.execute_api_call(endpoint["name"], **api_params)
            
            self.execution_stats["successful_calls"] += 1
            
            # Format successful response
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(result, indent=2, default=str)
                    )
                ]
            )
            
        except Exception as e:
            self.logger.error(f"MCP tool error for {name}: {str(e)}")
            self.execution_stats["failed_calls"] += 1
            
            # Format error response
            error_response = {
                "error": str(e),
                "tool": name,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(error_response, indent=2)
                    )
                ],
                isError=True
            )
    
    def _convert_mcp_to_api_params(
        self, 
        mcp_args: Dict[str, Any], 
        endpoint: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert MCP tool arguments to API parameters"""
        api_params = {}
        
        # Map MCP parameter names to API parameter names
        for param_name, param_value in mcp_args.items():
            # Convert snake_case to camelCase if needed
            camel_name = self._to_camel_case(param_name)
            
            # Check if parameter exists in endpoint definition
            if param_name in endpoint.get("parameters", {}):
                api_params[param_name] = param_value
            elif camel_name in endpoint.get("parameters", {}):
                api_params[camel_name] = param_value
            else:
                # Pass through as-is
                api_params[param_name] = param_value
        
        return api_params
    
    def _to_camel_case(self, snake_str: str) -> str:
        """Convert snake_case to camelCase"""
        components = snake_str.split('_')
        return components[0] + ''.join(x.title() for x in components[1:])
    
    async def list_resources(self) -> ListResourcesResult:
        """List available resources"""
        if not MCP_AVAILABLE:
            return {"resources": []}
        
        resources = [
            Resource(
                uri="bancs://api-documentation",
                name="TCS BAnCS API Documentation",
                description="Complete API documentation with examples",
                mimeType="text/markdown"
            ),
            Resource(
                uri="bancs://mcp-tools",
                name="MCP Tools List",
                description="List of all available MCP tools with schemas",
                mimeType="application/json"
            ),
            Resource(
                uri="bancs://cookbook-recipes",
                name="API Cookbook Recipes",
                description="Collection of API usage recipes",
                mimeType="application/json"
            ),
            Resource(
                uri="bancs://openapi-spec",
                name="OpenAPI Specification",
                description="OpenAPI 3.0 specification for all endpoints",
                mimeType="application/json"
            ),
            Resource(
                uri="bancs://postman-collection",
                name="Postman Collection",
                description="Postman collection for API testing",
                mimeType="application/json"
            )
        ]
        
        return ListResourcesResult(resources=resources)
    
    async def read_resource(self, uri: str) -> ReadResourceResult:
        """Read a specific resource"""
        if not MCP_AVAILABLE:
            return {"content": [{"type": "text", "text": "Resource not available"}]}
        
        self.logger.info(f"Reading resource: {uri}")
        
        try:
            if uri == "bancs://api-documentation":
                content = self._get_api_documentation()
            elif uri == "bancs://mcp-tools":
                content = self._get_mcp_tools_json()
            elif uri == "bancs://cookbook-recipes":
                content = self._get_cookbook_recipes()
            elif uri == "bancs://openapi-spec":
                content = self._get_openapi_spec()
            elif uri == "bancs://postman-collection":
                content = self._get_postman_collection()
            else:
                raise ValueError(f"Unknown resource: {uri}")
            
            return ReadResourceResult(
                content=[
                    TextContent(
                        type="text",
                        text=content
                    )
                ]
            )
        except Exception as e:
            self.logger.error(f"Error reading resource {uri}: {str(e)}")
            return ReadResourceResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Error reading resource: {str(e)}"
                    )
                ]
            )
    
    def _get_api_documentation(self) -> str:
        """Generate comprehensive API documentation"""
        doc = "# TCS BAnCS API Documentation\n\n"
        doc += f"Generated: {datetime.utcnow().isoformat()}\n\n"
        
        # Add table of contents
        doc += "## Table of Contents\n\n"
        for category in self.kb.api_categories.keys():
            doc += f"- [{category.title()} Management](#{category})\n"
        doc += "\n---\n\n"
        
        # Add endpoints by category
        for category, endpoints in self.kb.api_categories.items():
            if not endpoints:
                continue
            
            doc += f"## {category.title()} Management\n\n"
            
            for ep_name in endpoints:
                ep = self.kb.endpoints.get(ep_name)
                if not ep:
                    continue
                
                doc += f"### {ep_name}\n\n"
                doc += f"**Description**: {ep['description']}\n\n"
                doc += f"- **Method**: `{ep['method']}`\n"
                doc += f"- **URL**: `{ep['url']}`\n"
                doc += f"- **Category**: {ep.get('category', 'general')}\n"
                doc += f"- **Difficulty**: {ep.get('difficulty', 'intermediate')}\n"
                
                if ep.get('mcp_enabled'):
                    doc += f"- **MCP Tool**: `{ep.get('mcp_name')}`\n"
                
                # Add parameters
                if ep.get('parameters'):
                    doc += "\n**Parameters**:\n\n"
                    doc += "| Name | Type | Required | Description |\n"
                    doc += "|------|------|----------|-------------|\n"
                    
                    for param_name, param_info in ep['parameters'].items():
                        required = "Yes" if param_info.get('required') else "No"
                        doc += f"| {param_name} | {param_info.get('type', 'string')} | {required} | {param_info.get('description', '')} |\n"
                
                # Add example response if available
                if ep.get('response_example'):
                    doc += "\n**Example Response**:\n```json\n"
                    doc += json.dumps(ep['response_example'], indent=2)
                    doc += "\n```\n"
                
                doc += "\n---\n\n"
        
        return doc
    
    def _get_mcp_tools_json(self) -> str:
        """Get MCP tools as JSON"""
        tools_data = []
        
        for tool in self.kb.mcp_tools:
            tool_info = {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema
            }
            
            # Add usage example if available
            endpoint = self.kb.get_endpoint_by_mcp_name(tool.name)
            if endpoint:
                tool_info["endpoint"] = endpoint["name"]
                tool_info["method"] = endpoint["method"]
                tool_info["url"] = endpoint["url"]
            
            tools_data.append(tool_info)
        
        return json.dumps({
            "total": len(tools_data),
            "tools": tools_data,
            "server": {
                "name": settings.mcp_server_name,
                "port": settings.mcp_port,
                "version": settings.app_version
            }
        }, indent=2)
    
    def _get_cookbook_recipes(self) -> str:
        """Get cookbook recipes as JSON"""
        recipes = []
        
        # Create recipes from endpoints
        for name, endpoint in self.kb.endpoints.items():
            recipe = {
                "id": name.lower().replace(" ", "_"),
                "title": name,
                "description": endpoint["description"],
                "category": endpoint.get("category", "general"),
                "difficulty": endpoint.get("difficulty", "intermediate"),
                "method": endpoint["method"],
                "url": endpoint["url"],
                "tags": endpoint.get("tags", []),
                "mcp_enabled": endpoint.get("mcp_enabled", False)
            }
            
            if endpoint.get("mcp_enabled"):
                recipe["mcp_tool"] = endpoint.get("mcp_name")
            
            recipes.append(recipe)
        
        # Group by category
        categorized = {}
        for recipe in recipes:
            category = recipe["category"]
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(recipe)
        
        return json.dumps({
            "total": len(recipes),
            "categories": list(categorized.keys()),
            "recipes": categorized
        }, indent=2)
    
    def _get_openapi_spec(self) -> str:
        """Generate OpenAPI 3.0 specification"""
        openapi = {
            "openapi": "3.0.0",
            "info": {
                "title": "TCS BAnCS API",
                "version": settings.app_version,
                "description": "TCS BAnCS Banking API with MCP Support"
            },
            "servers": [
                {
                    "url": settings.api_base_url,
                    "description": "BAnCS API Server"
                }
            ],
            "paths": {},
            "components": {
                "securitySchemes": {
                    "ApiKeyAuth": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-API-Key"
                    }
                },
                "schemas": {}
            }
        }
        
        # Add paths
        for name, endpoint in self.kb.endpoints.items():
            path = endpoint["url"]
            method = endpoint["method"].lower()
            
            if path not in openapi["paths"]:
                openapi["paths"][path] = {}
            
            operation = {
                "summary": endpoint["description"],
                "operationId": name,
                "tags": [endpoint.get("category", "default")],
                "parameters": [],
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
            for param_name, param_info in endpoint.get("parameters", {}).items():
                param_in = "path" if "{" + param_name + "}" in endpoint["url"] else "query"
                operation["parameters"].append({
                    "name": param_name,
                    "in": param_in,
                    "required": param_info.get("required", False),
                    "schema": {
                        "type": param_info.get("type", "string")
                    },
                    "description": param_info.get("description", "")
                })
            
            # Add MCP extension
            if endpoint.get("mcp_enabled"):
                operation["x-mcp-tool"] = endpoint.get("mcp_name")
            
            openapi["paths"][path][method] = operation
        
        return json.dumps(openapi, indent=2)
    
    def _get_postman_collection(self) -> str:
        """Generate Postman collection"""
        collection = {
            "info": {
                "name": "TCS BAnCS API Collection",
                "description": "Generated Postman collection for TCS BAnCS APIs",
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
            },
            "item": []
        }
        
        # Group by category
        categories = {}
        for name, endpoint in self.kb.endpoints.items():
            category = endpoint.get("category", "general")
            if category not in categories:
                categories[category] = {
                    "name": category.title(),
                    "item": []
                }
            
            # Create request item
            request_item = {
                "name": name,
                "request": {
                    "method": endpoint["method"],
                    "header": [
                        {"key": "entity", "value": settings.bancs_entity},
                        {"key": "userId", "value": settings.bancs_user_id},
                        {"key": "languageCode", "value": settings.bancs_language_code}
                    ],
                    "url": {
                        "raw": settings.api_base_url + endpoint["url"],
                        "protocol": "https",
                        "host": ["demoapps", "tcsbancs", "com"],
                        "path": endpoint["url"].split("/")[1:]
                    }
                }
            }
            
            categories[category]["item"].append(request_item)
        
        collection["item"] = list(categories.values())
        
        return json.dumps(collection, indent=2)
    
    async def list_prompts(self) -> ListPromptsResult:
        """List available prompts for common tasks"""
        if not MCP_AVAILABLE:
            return {"prompts": []}
        
        prompts = [
            Prompt(
                name="get_balance",
                description="Get account balance for a specific account",
                arguments=[
                    {
                        "name": "account_reference",
                        "description": "15-digit account reference number",
                        "required": True
                    }
                ]
            ),
            Prompt(
                name="create_customer",
                description="Create a new customer with basic information",
                arguments=[
                    {
                        "name": "customer_name",
                        "description": "Full name of the customer",
                        "required": True
                    },
                    {
                        "name": "email",
                        "description": "Customer email address",
                        "required": True
                    }
                ]
            ),
            Prompt(
                name="transfer_funds",
                description="Transfer funds between accounts",
                arguments=[
                    {
                        "name": "from_account",
                        "description": "Source account reference",
                        "required": True
                    },
                    {
                        "name": "to_account",
                        "description": "Destination account reference",
                        "required": True
                    },
                    {
                        "name": "amount",
                        "description": "Transfer amount",
                        "required": True
                    }
                ]
            )
        ]
        
        return ListPromptsResult(prompts=prompts)
    
    async def get_prompt(self, name: str, arguments: Dict[str, Any]) -> GetPromptResult:
        """Get a specific prompt with filled arguments"""
        if not MCP_AVAILABLE:
            return {"messages": []}
        
        # Define prompt templates
        prompt_templates = {
            "get_balance": f"""Please retrieve the account balance for account reference: {arguments.get('account_reference', 'ACCOUNT_REF')}.
            
Use the get_account_balance tool to fetch the current balance, available balance, and any blocked amounts.""",
            
            "create_customer": f"""Create a new customer with the following information:
- Name: {arguments.get('customer_name', 'CUSTOMER_NAME')}
- Email: {arguments.get('email', 'EMAIL')}

Use the create_customer tool to register this customer in the system.""",
            
            "transfer_funds": f"""Transfer funds with these details:
- From Account: {arguments.get('from_account', 'FROM_ACCOUNT')}
- To Account: {arguments.get('to_account', 'TO_ACCOUNT')}
- Amount: {arguments.get('amount', 'AMOUNT')}

Use the create_transaction tool to process this transfer."""
        }
        
        prompt_text = prompt_templates.get(name, "Unknown prompt")
        
        return GetPromptResult(
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=prompt_text
                    )
                )
            ]
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get MCP server statistics"""
        return {
            "server_name": settings.mcp_server_name,
            "total_tools": len(self.kb.mcp_tools),
            "execution_stats": self.execution_stats,
            "uptime": "N/A",  # Could track actual uptime
            "version": settings.app_version
        }

async def run_mcp_server(server: IntegratedMCPServer):
    """Run the MCP server"""
    if not MCP_AVAILABLE:
        logger.warning("MCP package not available. MCP server not started.")
        return
    
    try:
        logger.info(f"Starting MCP server: {settings.mcp_server_name}")
        
        # In a real implementation, this would start the actual MCP server
        # For now, we'll keep it running
        while True:
            await asyncio.sleep(60)
            
    except Exception as e:
        logger.error(f"MCP server error: {str(e)}")

def create_standalone_mcp_server():
    """Create a standalone MCP server instance"""
    if not MCP_AVAILABLE:
        logger.error("MCP package not installed. Cannot create server.")
        return None
    
    from knowledge_base import UnifiedAPIKnowledgeBase
    
    # Initialize knowledge base
    kb = UnifiedAPIKnowledgeBase()
    
    # Create MCP server
    server = IntegratedMCPServer(kb)
    
    return server

async def main():
    """Main entry point for standalone MCP server"""
    if not MCP_AVAILABLE:
        print("Error: MCP package not installed.")
        print("Install with: pip install mcp")
        return
    
    server = create_standalone_mcp_server()
    if server:
        # Run with stdio transport
        async with stdio.stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())