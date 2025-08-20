# backend/response_generator.py
"""
Response Generator for TCS BAnCS API Chatbot
Generates intelligent responses based on queries and context
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Use relative import
from .config import settings

logger = logging.getLogger(__name__)

class UnifiedResponseGenerator:
    """Generates responses for both chatbot and MCP queries"""
    
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.logger = logging.getLogger(__name__)
        self.response_cache = {}
        self.cache_ttl = settings.cache_ttl if hasattr(settings, 'cache_ttl') else 300
        
        # Response templates
        self.templates = self._load_response_templates()
    
    def _load_response_templates(self) -> Dict[str, str]:
        """Load response templates for various query types"""
        return {
            "greeting": "Hello! I'm your TCS BAnCS API assistant.",
            "not_found": "I couldn't find specific information about that.",
            "error": "An error occurred while processing your request."
        }
    
    async def generate_chatbot_response(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate response for chatbot query"""
        # Simple response for now
        relevant_endpoints = self.kb.search_endpoints(query)
        
        if relevant_endpoints:
            endpoint_name = relevant_endpoints[0][0]
            endpoint = self.kb.endpoints.get(endpoint_name)
            
            response = f"I found information about {endpoint_name}:\n"
            response += f"Description: {endpoint.get('description', 'N/A')}\n"
            response += f"Method: {endpoint.get('method', 'N/A')}\n"
            response += f"URL: {endpoint.get('url', 'N/A')}"
            
            confidence = relevant_endpoints[0][1] / 100.0
        else:
            response = self.templates["not_found"]
            confidence = 0.0
        
        return {
            "response": response,
            "confidence": confidence,
            "mcp_available": settings.mcp_enabled,
            "api_references": [
                {"name": name, "score": score}
                for name, score in relevant_endpoints[:3]
            ]
        }

# ============= ADD THIS TO THE END OF backend/response_generator.py =============

    async def generate_response(self, message, context=None):
        """Generate response for chat message with MCP awareness"""
        message_lower = message.lower() if message else ""
        
        # Check for MCP/tool queries
        if 'mcp' in message_lower or 'tool' in message_lower:
            return await self._generate_mcp_response(message)
        
        # Search for relevant endpoints
        relevant_endpoints = self.kb.search_endpoints(message) if hasattr(self.kb, 'search_endpoints') else []
        
        response_text = ""
        code_examples = []
        
        # Handle specific query types
        if "balance" in message_lower or "account" in message_lower:
            response_text = """To get account balance, use the GET /account/balance endpoint.

    **Required Headers:**
    - entity: GPRDTTSTOU
    - userId: 1
    - languageCode: 1

    **Example Request:**
    ```python
    import requests

    response = requests.get(
        'https://demoapps.tcsbancs.com/Core/accountManagement/account/balanceDetails/{accountReference}',
        headers={
            'entity': 'GPRDTTSTOU',
            'userId': '1',
            'languageCode': '1'
        }
    )
    print(response.json())
    ```"""
            code_examples = [{
                "language": "python",
                "code": "# Get account balance\nresponse = requests.get(url, headers=headers)"
            }]
            
        elif "customer" in message_lower:
            response_text = """Customer management endpoints:

    **Available Endpoints:**
    - GET /customer/details - Get customer information
    - POST /customer - Create new customer
    - PUT /customer/{id} - Update customer
    - DELETE /customer/{id} - Delete customer

    All endpoints require authentication headers."""
            
        elif "auth" in message_lower or "header" in message_lower:
            response_text = """Authentication for BAnCS API requires these headers:

    **Required Headers:**
    ```json
    {
        "entity": "GPRDTTSTOU",
        "userId": "1",
        "languageCode": "1"
    }
    ```

    Include these headers in every API request for proper authentication."""
            
        elif "endpoint" in message_lower or "api" in message_lower or "available" in message_lower:
            # List available endpoints
            endpoints = await self.kb.list_endpoints() if hasattr(self.kb, 'list_endpoints') else []
            if endpoints:
                response_text = f"**Available API Endpoints ({len(endpoints)} total):**\n\n"
                
                # Group by category
                by_category = {}
                for ep in endpoints:
                    cat = ep.get('category', 'general')
                    if cat not in by_category:
                        by_category[cat] = []
                    by_category[cat].append(ep)
                
                for category, eps in list(by_category.items())[:5]:  # Show first 5 categories
                    response_text += f"**{category.title()}:**\n"
                    for ep in eps[:3]:  # Show first 3 endpoints per category
                        response_text += f"• {ep['method']} {ep['path']} - {ep.get('description', '')}\n"
                    response_text += "\n"
            else:
                response_text = "I can help you with API endpoints. Try asking about specific operations."
        
        else:
            # Default response with relevant endpoints if found
            if relevant_endpoints:
                endpoint_name = relevant_endpoints[0][0]
                endpoint = self.kb.endpoints.get(endpoint_name, {})
                response_text = f"""Found relevant endpoint: **{endpoint_name}**

    **Description:** {endpoint.get('description', 'N/A')}
    **Method:** {endpoint.get('method', 'GET')}
    **Path:** {endpoint.get('url', '/')}

    Use the appropriate headers and parameters for this endpoint."""
            else:
                response_text = """I can help you with the TCS BAnCS API.

    **Available Resources:**
    - Account Management (balance, details, transactions)
    - Customer Management (create, update, retrieve)
    - Loan Management (details, payments)
    - Transaction Processing (transfers, bookings)
    - MCP Tools (for AI integration)

    What specific operation would you like to know about?"""
        
        return {
            "response": response_text,
            "code_examples": code_examples,
            "confidence": 0.8 if relevant_endpoints or any(keyword in message_lower for keyword in ['balance', 'customer', 'auth', 'mcp']) else 0.5,
            "mcp_available": hasattr(self.kb, 'mcp_tools') and len(self.kb.mcp_tools) > 0,
            "api_references": [
                {"name": name, "score": score}
                for name, score in (relevant_endpoints[:3] if relevant_endpoints else [])
            ]
        }

    async def _generate_mcp_response(self, message):
        """Generate response specifically about MCP tools"""
        mcp_tools = []
        
        if hasattr(self.kb, 'mcp_tools'):
            for tool in self.kb.mcp_tools:
                mcp_tools.append({
                    'name': getattr(tool, 'name', 'unknown'),
                    'description': getattr(tool, 'description', '')
                })
        
        if mcp_tools:
            response = f"**MCP (Model Context Protocol) Tools Available: {len(mcp_tools)}**\n\n"
            response += "These tools allow AI models to interact with the BAnCS API:\n\n"
            
            for tool in mcp_tools[:10]:  # Show first 10
                response += f"• **{tool['name']}**: {tool['description']}\n"
            
            if len(mcp_tools) > 10:
                response += f"\n...and {len(mcp_tools) - 10} more tools.\n"
            
            response += "\n**Usage:** These tools enable direct API calls through the MCP protocol."
        else:
            response = "MCP tools are configured but none are currently loaded. Check the API configuration."
        
        return {
            "response": response,
            "code_examples": [],
            "confidence": 1.0,
            "mcp_available": True,
            "api_references": []
        }
