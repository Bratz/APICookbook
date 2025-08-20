"""
Integration layer for BERT chatbot with the API Cookbook
Fixed version with proper error handling and missing methods
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import asyncio
import logging

logger = logging.getLogger(__name__)

# Use try/except for imports to handle different contexts
try:
    from .model import BERTCookbookChatbot
except ImportError:
    from model import BERTCookbookChatbot

try:
    from ..knowledge_base import UnifiedAPIKnowledgeBase
    from ..response_generator import UnifiedResponseGenerator
except ImportError:
    from knowledge_base import UnifiedAPIKnowledgeBase
    from response_generator import UnifiedResponseGenerator

class BERTChatbotIntegration:
    """
    Integrates BERT chatbot with existing cookbook infrastructure
    """
    
    def __init__(self, knowledge_base: UnifiedAPIKnowledgeBase):
        self.kb = knowledge_base
        self.bert_chatbot = BERTCookbookChatbot()
        self.response_generator = UnifiedResponseGenerator(knowledge_base)
        
        # Load API knowledge into BERT model
        self._sync_knowledge_base()
    
    def _sync_knowledge_base(self):
        """Sync knowledge base with BERT model"""
        synced_count = 0
        
        # Convert KB endpoints to BERT format
        for endpoint_name, endpoint in self.kb.endpoints.items():
            # Convert parameters from dict to list format
            parameters_list = []
            parameters = endpoint.get('parameters', {})
            
            if isinstance(parameters, dict):
                # Convert dict format to list format
                for param_name, param_info in parameters.items():
                    param_dict = {'name': param_name}
                    if isinstance(param_info, dict):
                        param_dict.update(param_info)
                    else:
                        # Handle case where param_info is just a type string
                        param_dict['type'] = str(param_info)
                    parameters_list.append(param_dict)
            elif isinstance(parameters, list):
                # Already in list format
                parameters_list = parameters
            
            self.bert_chatbot.api_metadata[endpoint_name] = {
                'method': endpoint.get('method', 'GET'),
                'path': endpoint.get('url', '/'),
                'parameters': parameters_list,  # Use the converted list
                'description': endpoint.get('description', '')
            }
            
            # Generate embeddings
            description = f"{endpoint.get('description', '')} {endpoint_name}"
            try:
                embeddings = self.bert_chatbot._generate_embeddings(description)
                self.bert_chatbot.api_embeddings[endpoint_name] = embeddings
                synced_count += 1
            except Exception as e:
                logger.warning(f"Failed to generate embeddings for {endpoint_name}: {e}")
        
        logger.info(f"Synced {synced_count} endpoints to BERT model")
    
    async def process_developer_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process developer query using BERT model
        """
        try:
            # Add debug logging
            logger.info(f"BERT processing query: {query}")
            logger.info(f"Knowledge base has {len(self.kb.endpoints)} endpoints")
            
            # Use BERT model for understanding
            bert_response = await self.bert_chatbot.process_query(query, context)
            
            logger.info(f"BERT response intent: {bert_response.get('intent')}")
            logger.info(f"BERT response confidence: {bert_response.get('confidence')}")
            
            # Enhance with knowledge base
            enhanced_response = await self._enhance_response(bert_response, query)
            
            # Format for UI
            formatted_response = self._format_for_ui(enhanced_response)
            
            logger.info(f"Formatted response sections: {len(formatted_response.get('sections', []))}")
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"BERT processing error: {e}", exc_info=True)
            # Return a more helpful error response
            return {
                'status': 'error',
                'message': f"I encountered an error processing your request: {str(e)}",
                'intent': 'error',
                'confidence': 0,
                'sections': [],
                'response': f"Error: {str(e)}"  # Add response field for compatibility
            }
    
    async def _enhance_response(self, bert_response: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Enhance BERT response with additional context"""
        
        # Get cookbook recipes if relevant
        if bert_response.get('intent') == 'get_example':
            try:
                recipes = await self.kb.get_cookbook_recipes()
                bert_response['cookbook_recipes'] = recipes
            except Exception as e:
                logger.warning(f"Could not get cookbook recipes: {e}")
        
        # Add MCP tool information
        if 'mcp' in query.lower() or 'tool' in query.lower():
            # Check if export_mcp_tools method exists
            if hasattr(self.kb, 'export_mcp_tools'):
                mcp_tools = self.kb.export_mcp_tools()
            else:
                # Fallback: get MCP tools directly if available
                mcp_tools = []
                if hasattr(self.kb, 'mcp_tools'):
                    for tool in self.kb.mcp_tools:
                        mcp_tools.append({
                            'name': getattr(tool, 'name', 'unknown'),
                            'description': getattr(tool, 'description', ''),
                        })
            bert_response['mcp_tools'] = mcp_tools
        
        return bert_response
    
    def _format_for_ui(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Format response for cookbook UI with proper code examples"""
        
        # Build the main message
        base_message = response.get('message', '')
        
        # If no message but we have other content, build one
        if not base_message:
            if response.get('code_examples'):
                base_message = "Here are code examples for your query:"
            elif response.get('error_analysis'):
                base_message = "I found an error analysis for your issue."
            elif response.get('related_apis'):
                base_message = f"I found {len(response.get('related_apis', []))} related API endpoints."
            else:
                # Default message based on intent
                intent_messages = {
                    'get_example': "Here are code examples for the API:",
                    'debug_error': "Let me help you debug this error:",
                    'find_api': "I found these relevant APIs:",
                    'understand_params': "Here are the parameter details:",
                    'best_practices': "Here are the best practices:",
                    'authentication': "For authentication, use these headers:\n- entity: GPRDTTSTOU\n- userId: 1\n- languageCode: 1"
                }
                base_message = intent_messages.get(
                    response.get('intent', ''),
                    "Here's what I found for your query:"
                )
        
        # Ensure we always have a message
        if not base_message:
            base_message = "I processed your query. See the details below."
        
        # BUILD COMPLETE MESSAGE WITH CODE
        complete_message = base_message
        
        # ALWAYS add code examples to the message for display
        if 'code_examples' in response and response['code_examples']:
            for i, example in enumerate(response['code_examples']):
                if isinstance(example, dict):
                    lang = example.get('language', 'python')
                    code = example.get('code', '')
                    description = example.get('description', '')
                    
                    # Add section header
                    if i == 0:
                        complete_message += "\n\n**Code Example:**"
                    else:
                        complete_message += f"\n\n**Example {i+1}:**"
                    
                    if description:
                        complete_message += f"\n{description}"
                    
                    # Add the code block
                    complete_message += f"\n```{lang}\n{code}\n```"
                    
                    # Add notes if present
                    if example.get('notes'):
                        complete_message += "\n\n**Notes:**"
                        for note in example['notes']:
                            complete_message += f"\n- {note}"
        
        # Add parameters section if present
        if 'parameters' in response and response['parameters']:
            complete_message += "\n\n**Parameters:**"
            for param in response['parameters']:
                required = "required" if param.get('required') else "optional"
                param_type = param.get('type', 'string')
                description = param.get('description', 'No description')
                complete_message += f"\n- `{param['name']}` ({param_type}, {required}): {description}"
        
        # Add authentication section if present and not already in message
        if 'authentication' in response and response['authentication']:
            auth = response['authentication']
            if auth.get('description'):
                complete_message += f"\n\n**Authentication:**\n{auth['description']}"
            
            # Add examples for different languages if present
            if auth.get('example'):
                for lang, example_code in auth['example'].items():
                    if example_code and lang in ['python', 'javascript', 'curl']:
                        complete_message += f"\n\n**{lang.title()} auth example:**"
                        if lang == 'curl':
                            complete_message += f"\n```bash\n{example_code}\n```"
                        else:
                            complete_message += f"\n```{lang}\n{example_code}\n```"
        
        # ADD RELATED APIS TO THE MESSAGE - THIS IS THE MISSING PART
        if 'related_apis' in response and response['related_apis']:
            complete_message += "\n\n**🔗 Related API Endpoints:**"
            for i, api in enumerate(response['related_apis'][:5], 1):  # Show top 5
                # Format each API nicely
                api_name = api.get('name', 'Unknown')
                method = api.get('method', 'GET')
                path = api.get('path', '/')
                confidence = api.get('confidence', 0)
                
                # Add the API with formatting
                complete_message += f"\n\n{i}. **{api_name}**"
                complete_message += f"\n   `{method} {path}`"
                
                # Add confidence if it's less than 1 (not an exact match)
                if confidence < 0.99:
                    complete_message += f"\n   Match confidence: {confidence:.1%}"
                
                # Add description if available
                if api.get('description'):
                    complete_message += f"\n   {api['description']}"
        
        # Add suggestions at the end
        if 'suggestions' in response and response['suggestions']:
            complete_message += "\n\n**💡 Next Steps:**"
            for suggestion in response['suggestions']:
                if suggestion:  # Check for None values
                    complete_message += f"\n• {suggestion}"
        
        # Format the response
        formatted = {
            'status': response.get('status', 'success'),
            'message': complete_message,  # Use the complete message with code
            'response': complete_message,  # CRITICAL: Set both fields to complete message
            'intent': response.get('intent', 'unknown'),
            'confidence': response.get('confidence', 0.5),
            'sections': [],
            'code_examples': response.get('code_examples', []),  # Preserve structured data
            'api_references': response.get('api_references', []),
            'parameters': response.get('parameters', []),
            'authentication': response.get('authentication', {}),
            'related_apis': response.get('related_apis', [])  # IMPORTANT: Include at top level
        }
        
        # Add sections for structured display (keeping your existing section logic)
        if 'code_examples' in response and response['code_examples']:
            formatted['sections'].append({
                'type': 'code_examples',
                'title': 'Code Examples',
                'content': response['code_examples']
            })
        
        if 'error_analysis' in response and response['error_analysis']:
            formatted['sections'].append({
                'type': 'error_analysis',
                'title': 'Error Analysis',
                'content': response['error_analysis']
            })
        
        if 'suggestions' in response and response['suggestions']:
            formatted['sections'].append({
                'type': 'suggestions',
                'title': 'Next Steps',
                'content': response['suggestions']
            })
        
        if 'related_apis' in response and response['related_apis']:
            formatted['sections'].append({
                'type': 'related_apis',
                'title': 'Related APIs',
                'content': response['related_apis']
            })
        
        logger.info(f"[BERT] Formatted response with {len(complete_message)} chars, {len(formatted.get('code_examples', []))} code examples, {len(formatted['sections'])} sections")
        logger.debug(f"[BERT] Complete message preview: {complete_message[:200]}...")
        logger.debug(f"[BERT] Related APIs in response: {len(response.get('related_apis', []))} items")
        
        return formatted


# FastAPI endpoints for BERT chatbot
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/bert-chat", tags=["BERT Chatbot"])

class BERTChatRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    preferences: Optional[Dict[str, str]] = None

class BERTChatResponse(BaseModel):
    status: str
    message: str
    response: Optional[str] = None  # Add for compatibility
    intent: Optional[str] = None
    confidence: float
    sections: List[Dict[str, Any]]

@router.post("/query", response_model=BERTChatResponse)
async def process_bert_query(request: BERTChatRequest, req: Request):
    """Process developer query using BERT model"""
    try:
        # Get integration instance from app state
        if not hasattr(req.app.state, 'bert_integration'):
            raise HTTPException(
                status_code=503,
                detail="BERT integration not initialized"
            )
        
        integration = req.app.state.bert_integration
        
        # Update user preferences if provided
        if request.preferences:
            integration.bert_chatbot.user_preferences.update(request.preferences)
        
        # Process query
        response = await integration.process_developer_query(
            query=request.query,
            context=request.context,
            session_id=request.session_id
        )
        
        # Ensure response field exists
        if 'response' not in response:
            response['response'] = response.get('message', '')
        
        return BERTChatResponse(**response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"BERT chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/intents")
async def get_supported_intents():
    """Get list of supported intents"""
    return {
        "intents": [
            {
                "name": "get_example",
                "description": "Request code examples",
                "example_queries": [
                    "Show me Python code for account balance",
                    "How do I call this API in JavaScript?"
                ]
            },
            {
                "name": "debug_error",
                "description": "Debug API errors",
                "example_queries": [
                    "I'm getting 401 error",
                    "Why is my request failing?"
                ]
            },
            {
                "name": "find_api",
                "description": "Find relevant APIs",
                "example_queries": [
                    "API for creating customer",
                    "How to get account transactions?"
                ]
            },
            {
                "name": "best_practices",
                "description": "Best practices and patterns",
                "example_queries": [
                    "Best practices for error handling",
                    "How to implement retry logic?"
                ]
            }
        ]
    }

@router.post("/feedback")
async def submit_feedback(
    session_id: str,
    query: str,
    response: Dict[str, Any],
    rating: int,
    comments: Optional[str] = None
):
    """Submit feedback for improving the model"""
    try:
        # Store feedback for model improvement
        feedback_data = {
            "session_id": session_id,
            "query": query,
            "response": response,
            "rating": rating,
            "comments": comments,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Create training_data directory if it doesn't exist
        feedback_dir = Path("training_data")
        feedback_dir.mkdir(exist_ok=True)
        
        # Save to feedback log for retraining
        feedback_path = feedback_dir / "feedback.jsonl"
        with open(feedback_path, 'a') as f:
            f.write(json.dumps(feedback_data) + '\n')
        
        return {"status": "success", "message": "Feedback recorded"}
    
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")


# Add the missing export_mcp_tools method to UnifiedAPIKnowledgeBase
def add_export_mcp_tools_to_kb(kb_class):
    """
    Monkey-patch to add export_mcp_tools method if it doesn't exist
    This should be added to the actual knowledge_base.py file
    """
    if not hasattr(kb_class, 'export_mcp_tools'):
        def export_mcp_tools(self):
            """Export MCP tools information"""
            tools = []
            if hasattr(self, 'mcp_tools'):
                for tool in self.mcp_tools:
                    tool_info = {
                        'name': getattr(tool, 'name', 'unknown'),
                        'description': getattr(tool, 'description', ''),
                    }
                    if hasattr(tool, 'inputSchema'):
                        tool_info['inputSchema'] = tool.inputSchema
                    tools.append(tool_info)
            return tools
        
        kb_class.export_mcp_tools = export_mcp_tools