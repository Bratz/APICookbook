"""
Sentence Transformer-based chatbot model for API Cookbook
Lightweight alternative to BERT - no Hugging Face downloads needed
Uses the unified knowledge base instead of loading separately
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from pathlib import Path
import logging
import asyncio
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

logger = logging.getLogger(__name__)

@dataclass
class QueryIntent:
    """Represents the intent of a developer query"""
    primary_intent: str
    confidence: float
    entities: Dict[str, Any]
    context_required: bool

@dataclass
class APIMatch:
    """Represents a matched API endpoint"""
    endpoint_name: str
    method: str
    path: str
    confidence: float
    parameters: List[Dict[str, Any]]

class SentenceTransformerChatbot:
    """
    Lightweight chatbot using Sentence Transformers approach
    No heavy model downloads required - uses TF-IDF and embeddings
    """
    
    def __init__(self, knowledge_base=None):
        """Initialize with TF-IDF vectorizer and use the shared knowledge base"""
        
        # Store reference to the knowledge base
        self.kb = knowledge_base
        
        # Use TF-IDF for text embeddings (no model download needed)
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
            use_idf=True,
            sublinear_tf=True
        )
        
        # Pre-fit on common API documentation terms
        self._initialize_vectorizer()
        
        # Initialize components without neural networks
        self.intent_classifier = self._build_intent_classifier()
        self.entity_extractor = self._build_entity_extractor()
        self.code_generator = self._build_code_generator()
        self.error_analyzer = self._build_error_analyzer()
        
        # API knowledge storage
        self.api_embeddings = {}
        self.api_metadata = {}
        self.api_texts = {}  # Store original texts for TF-IDF
        
        # Index APIs from knowledge base if provided
        if self.kb:
            self.index_from_knowledge_base()
        else:
            # Fallback to loading from files
            self.load_api_knowledge()
        
        # Conversation context
        self.conversation_history = []
        self.user_preferences = {
            'preferred_language': 'python',
            'experience_level': 'intermediate',
            'verbosity': 'normal'
        }
        
        self.current_query = ""
    
    def _initialize_vectorizer(self):
        """Initialize TF-IDF with common API terms"""
        # Common API documentation terms to bootstrap the vectorizer
        api_terms = [
            "get post put delete patch api endpoint",
            "request response header body parameter",
            "authentication authorization token bearer",
            "json xml content type accept",
            "error status code success failure",
            "create read update delete crud",
            "account customer transaction balance loan",
            "mobile number email address phone",
            "query path body header cookie",
            "required optional default example",
            "string integer boolean array object",
            "rate limit throttle quota exceeded",
            "validation schema format pattern",
            "webhook callback async sync",
            "pagination offset limit cursor",
            "customer management account management",
            "transaction processing payment transfer"
        ]
        
        # Fit the vectorizer on common terms
        self.vectorizer.fit(api_terms)
    
    def index_from_knowledge_base(self):
        """Index APIs from the unified knowledge base"""
        if not self.kb or not hasattr(self.kb, 'endpoints'):
            logger.warning("Knowledge base not available or has no endpoints")
            return
        
        # Index all endpoints from the knowledge base
        for endpoint_name, endpoint_data in self.kb.endpoints.items():
            # Build text representation for TF-IDF
            text_parts = []
            
            # Add basic info
            text_parts.append(endpoint_data.get('summary', ''))
            text_parts.append(endpoint_data.get('description', ''))
            text_parts.append(endpoint_data.get('url', '').replace('/', ' ').replace('{', ' ').replace('}', ' '))
            text_parts.append(endpoint_name.replace('_', ' ').replace('-', ' '))
            
            # Add parameters
            parameters = endpoint_data.get('parameters', {})
            if isinstance(parameters, dict):
                for param_name, param_info in parameters.items():
                    text_parts.append(param_name)
                    if isinstance(param_info, dict):
                        text_parts.append(param_info.get('description', ''))
            
            # Add tags/categories
            text_parts.append(endpoint_data.get('category', ''))
            text_parts.extend(endpoint_data.get('tags', []))
            
            # Combine all text
            full_text = ' '.join(filter(None, text_parts))
            
            # Create a standardized key
            method = endpoint_data.get('method', 'GET')
            path = endpoint_data.get('url', '/')
            endpoint_key = f"{method} {path}"
            
            self.api_texts[endpoint_key] = full_text
            
            # Store metadata
            self.api_metadata[endpoint_key] = {
                'method': method,
                'path': path,
                'parameters': parameters,
                'description': endpoint_data.get('description', ''),
                'summary': endpoint_data.get('summary', ''),
                'category': endpoint_data.get('category', ''),
                'tags': endpoint_data.get('tags', []),
                'full_context': full_text,
                'original_name': endpoint_name
            }
        
        # Create TF-IDF embeddings for all APIs
        if self.api_texts:
            texts = list(self.api_texts.values())
            try:
                # Re-fit vectorizer on actual API texts
                self.vectorizer.fit(texts)
                embeddings = self.vectorizer.transform(texts)
                
                # Store embeddings
                for i, key in enumerate(self.api_texts.keys()):
                    self.api_embeddings[key] = embeddings[i]
                    
                logger.info(f"Indexed {len(self.api_texts)} API endpoints from knowledge base")
            except Exception as e:
                logger.warning(f"Error creating TF-IDF embeddings: {e}")
    
    def _build_intent_classifier(self):
        """Build keyword-based intent classifier"""
        class IntentClassifier:
            def __init__(self):
                self.intent_patterns = {
                    'list_endpoints': {
                        'keywords': ['list', 'show', 'all', 'available', 'what apis', 'show me the apis', 'get me', 'show apis'],
                        'weight': 1.2
                    },
                    'get_example': {
                        'keywords': ['code', 'example', 'sample', 'snippet', 'how to', 'implement', 'usage', 'python', 'javascript'],
                        'weight': 1.0
                    },
                    'find_api': {
                        'keywords': ['find', 'search', 'which api', 'endpoint for', 'customer', 'account', 'transaction', 'management', 'loan', 'balance'],
                        'weight': 0.9
                    },
                    'debug_error': {
                        'keywords': ['error', '401', '403', '404', '500', 'fail', 'wrong', 'issue', 'not working', 'broken'],
                        'weight': 0.95
                    },
                    'understand_params': {
                        'keywords': ['parameter', 'param', 'argument', 'input', 'field', 'required', 'optional'],
                        'weight': 0.85
                    },
                    'authentication': {
                        'keywords': ['auth', 'token', 'header', 'credential', 'login', 'api key', 'entity', 'userid'],
                        'weight': 0.9
                    },
                    'greeting': {
                        'keywords': ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon'],
                        'weight': 1.0
                    }
                }
                
                # Labels for compatibility
                self.intent_labels = list(self.intent_patterns.keys())
            
            def classify(self, query: str) -> Tuple[str, float]:
                """Classify intent using keyword matching"""
                query_lower = query.lower().strip()
                
                # Check for exact phrase matches first
                if query_lower in ['hi', 'hello', 'hey']:
                    return 'greeting', 0.95
                    
                if 'show me the apis' in query_lower or 'list all' in query_lower or 'show all' in query_lower:
                    return 'list_endpoints', 0.95
                    
                scores = {}
                
                for intent, config in self.intent_patterns.items():
                    score = 0
                    for keyword in config['keywords']:
                        if keyword in query_lower:
                            # Give higher score for longer keyword matches
                            score += config['weight'] * (len(keyword) / 10)
                    scores[intent] = score
                
                if not scores or max(scores.values()) == 0:
                    # Better default based on query patterns
                    if any(word in query_lower for word in ['show', 'list', 'get', 'all']):
                        return 'list_endpoints', 0.6
                    return 'find_api', 0.5  # Default intent
                
                best_intent = max(scores, key=scores.get)
                confidence = min(scores[best_intent], 1.0)
                
                return best_intent, confidence
        
        return IntentClassifier()
    
    def _build_entity_extractor(self):
        """Build regex-based entity extractor"""
        class EntityExtractor:
            def __init__(self):
                self.entity_types = [
                    'API_ENDPOINT',
                    'HTTP_METHOD',
                    'PARAMETER_NAME',
                    'ERROR_CODE',
                    'PROGRAMMING_LANGUAGE',
                    'DATA_TYPE',
                    'RESOURCE_ID',
                    'FIELD_NAME'
                ]
                
                self.patterns = {
                    'HTTP_METHOD': r'\b(GET|POST|PUT|DELETE|PATCH)\b',
                    'ERROR_CODE': r'\b[4-5]\d{2}\b',
                    'PROGRAMMING_LANGUAGE': r'\b(python|javascript|java|curl|c#|ruby|go|php)\b',
                    'API_ENDPOINT': r'\/[a-zA-Z0-9\/\-\_\{\}]+',
                    'PARAMETER_NAME': r'(?:param|parameter|field|arg)[:\s]+(\w+)',
                }
            
            def extract(self, query: str) -> Dict[str, List[str]]:
                """Extract entities using regex patterns"""
                entities = {}
                
                for entity_type, pattern in self.patterns.items():
                    matches = re.findall(pattern, query, re.IGNORECASE)
                    if matches:
                        entities[entity_type] = list(set(matches))
                
                return entities
        
        return EntityExtractor()
    
    def _build_code_generator(self):
        """Build template-based code generator"""
        class CodeGenerator:
            def __init__(self):
                self.available = True
                self.templates = self._load_templates()
            
            def _load_templates(self):
                """Load code templates"""
                return {
                    'python': {
                        'get': """import httpx

async def {func_name}({params}):
    headers = {{
        'entity': 'GPRDTTSTOU',
        'userId': '1',
        'languageCode': '1',
        'Content-Type': 'application/json'
    }}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "{url}",
            headers=headers{query_params}
        )
        return response.json()""",
                        'post': """import httpx

async def {func_name}({params}):
    headers = {{
        'entity': 'GPRDTTSTOU',
        'userId': '1',
        'languageCode': '1',
        'Content-Type': 'application/json'
    }}
    
    data = {{{body}}}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "{url}",
            headers=headers,
            json=data
        )
        return response.json()"""
                    }
                }
            
            def generate(self, prompt: str, max_length: int = 256) -> str:
                """Generate code using templates"""
                # Simple template selection based on prompt
                if 'post' in prompt.lower():
                    template = self.templates.get('python', {}).get('post', '')
                else:
                    template = self.templates.get('python', {}).get('get', '')
                
                # Fill in basic template
                code = template.format(
                    func_name='api_call',
                    params='',
                    url='https://api.example.com/endpoint',
                    query_params='',
                    body=''
                )
                
                return code
        
        return CodeGenerator()
    
    def _build_error_analyzer(self):
        """Build rule-based error analyzer"""
        class ErrorAnalyzer:
            def __init__(self):
                self.error_categories = [
                    'authentication_error',
                    'rate_limit_exceeded',
                    'invalid_parameters',
                    'resource_not_found',
                    'permission_denied',
                    'server_error',
                    'network_error',
                    'parsing_error',
                    'validation_error',
                    'timeout_error'
                ]
                
                self.error_patterns = {
                    '401': 'authentication_error',
                    '403': 'permission_denied',
                    '404': 'resource_not_found',
                    '429': 'rate_limit_exceeded',
                    '400': 'invalid_parameters',
                    '500': 'server_error',
                    '502': 'network_error',
                    '503': 'server_error',
                    '504': 'timeout_error'
                }
            
            def analyze(self, error_code: str) -> str:
                """Analyze error code and return category"""
                return self.error_patterns.get(error_code, 'unknown_error')
        
        return ErrorAnalyzer()
    
    def load_api_knowledge(self):
        """Load and index API documentation (fallback when KB not available)"""
        api_specs_path = Path("api_specs")
        
        if api_specs_path.exists():
            for spec_file in api_specs_path.glob("*.json"):
                try:
                    with open(spec_file) as f:
                        spec = json.load(f)
                        self._index_api_spec(spec)
                except Exception as e:
                    logger.warning(f"Failed to load spec file {spec_file}: {e}")
    
    def _index_api_spec(self, spec: Dict[str, Any]):
        """Index API specification using TF-IDF"""
        for path, methods in spec.get('paths', {}).items():
            for method, details in methods.items():
                if method in ['get', 'post', 'put', 'delete', 'patch']:
                    endpoint_key = f"{method.upper()} {path}"
                    
                    # Build text representation for TF-IDF
                    text_parts = []
                    
                    # Add basic info
                    text_parts.append(details.get('summary', ''))
                    text_parts.append(details.get('description', ''))
                    text_parts.append(path.replace('/', ' ').replace('{', ' ').replace('}', ' '))
                    
                    # Add parameters
                    parameters = details.get('parameters', [])
                    for param in parameters:
                        if isinstance(param, dict):
                            text_parts.append(param.get('name', ''))
                            text_parts.append(param.get('description', ''))
                    
                    # Add tags
                    tags = details.get('tags', [])
                    text_parts.extend(tags)
                    
                    # Combine all text
                    full_text = ' '.join(filter(None, text_parts))
                    self.api_texts[endpoint_key] = full_text
                    
                    # Store metadata
                    self.api_metadata[endpoint_key] = {
                        'method': method.upper(),
                        'path': path,
                        'parameters': parameters,
                        'tags': tags,
                        'operation_id': details.get('operationId', ''),
                        'description': details.get('description', ''),
                        'summary': details.get('summary', ''),
                        'full_context': full_text
                    }
        
        # Create TF-IDF embeddings for all APIs
        if self.api_texts:
            texts = list(self.api_texts.values())
            try:
                # Re-fit vectorizer on actual API texts
                self.vectorizer.fit(texts)
                embeddings = self.vectorizer.transform(texts)
                
                # Store embeddings
                for i, key in enumerate(self.api_texts.keys()):
                    self.api_embeddings[key] = embeddings[i]
                    
                logger.info(f"Indexed {len(self.api_texts)} API endpoints")
            except Exception as e:
                logger.warning(f"Error creating TF-IDF embeddings: {e}")
    
    def _generate_embeddings(self, text: str):
        """Generate TF-IDF embeddings for text"""
        try:
            # Transform text to TF-IDF vector
            embedding = self.vectorizer.transform([text])
            return embedding
        except:
            # Return zero vector if transformation fails
            return np.zeros((1, self.vectorizer.max_features))
    
    async def process_query(
        self,
        query: str,
        context: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Process a developer query and generate response"""
        try:
            logger.info(f"[ST] Query: '{query}'")
            self.current_query = query
            
            # Get query embeddings
            query_embeddings = self._generate_embeddings(query)
            
            # Classify intent using keyword classifier
            intent_label, confidence = self.intent_classifier.classify(query)
            intent = QueryIntent(
                primary_intent=intent_label,
                confidence=confidence,
                entities={},
                context_required=intent_label in ['debug_error', 'compare_apis']
            )
            
            # Extract entities
            entities = self.entity_extractor.extract(query)
            intent.entities = entities
            
            logger.info(f"[ST] Classified intent: {intent.primary_intent} (confidence: {intent.confidence:.2f})")
            
            # Find relevant APIs
            relevant_apis = self._find_relevant_apis(query_embeddings, intent, entities)
            
            logger.info(f"[ST] Found {len(relevant_apis)} relevant APIs")
            
            if relevant_apis:
                top_match = relevant_apis[0]
                logger.info(f"[ST] Top match: {top_match.endpoint_name} (confidence: {top_match.confidence:.2f})")
            
            # Generate response based on intent
            response = {}
            
            # Handle greeting intent
            if intent.primary_intent == 'greeting':
                response = {
                    'status': 'success',
                    'intent': intent.primary_intent,
                    'confidence': intent.confidence,
                    'response': "Hello! I'm your TCS BaNCS API assistant. I can help you with:\n\n" +
                               "• Finding API endpoints\n" +
                               "• Generating code examples\n" +
                               "• Explaining authentication\n" +
                               "• Debugging API errors\n\n" +
                               "What would you like to know about the APIs?",
                    'message': "Welcome! How can I help you with the BaNCS APIs?"
                }
            
            # Handle list_endpoints intent
            elif intent.primary_intent == 'list_endpoints':
                # Use the knowledge base directly
                if self.kb and hasattr(self.kb, 'endpoints'):
                    all_endpoints = []
                    for endpoint_name, endpoint_data in self.kb.endpoints.items():
                        all_endpoints.append({
                            'name': endpoint_name,
                            'method': endpoint_data.get('method', 'GET'),
                            'path': endpoint_data.get('url', '/'),
                            'description': endpoint_data.get('description', '')[:200],
                            'category': endpoint_data.get('category', 'general'),
                            'tags': endpoint_data.get('tags', [])
                        })
                    
                    # Group by category
                    categories = {}
                    for ep in all_endpoints:
                        cat = ep['category']
                        if cat not in categories:
                            categories[cat] = []
                        categories[cat].append(ep)
                    
                    # Build response
                    response_text = f"I have access to **{len(all_endpoints)} API endpoints** across {len(categories)} categories:\n\n"
                    
                    # Show top categories with examples
                    for category, eps in list(categories.items())[:5]:
                        response_text += f"**{category.replace('_', ' ').title()}** ({len(eps)} endpoints):\n"
                        for ep in eps[:3]:
                            response_text += f"  • `{ep['method']} {ep['path']}`\n"
                        if len(eps) > 3:
                            response_text += f"  _...and {len(eps) - 3} more_\n"
                        response_text += "\n"
                    
                    if len(categories) > 5:
                        response_text += f"_...and {len(categories) - 5} more categories_\n\n"
                    
                    response_text += "You can ask me about specific endpoints or categories for more details!"
                    
                    response = {
                        'status': 'success',
                        'intent': intent.primary_intent,
                        'confidence': intent.confidence,
                        'response': response_text,
                        'message': f"Found {len(all_endpoints)} total API endpoints",
                        'endpoints': all_endpoints[:20],
                        'total_count': len(all_endpoints),
                        'categories': list(categories.keys())
                    }
                else:
                    # Fallback if KB not available
                    response = {
                        'status': 'success',
                        'intent': intent.primary_intent,
                        'confidence': 0.5,
                        'response': "I'm having trouble accessing the API list. Please try again or ask about specific endpoints.",
                        'message': "Unable to retrieve endpoint list"
                    }
            
            elif intent.primary_intent == 'get_example':
                if relevant_apis:
                    code_examples_dict = self._generate_code_example(relevant_apis, entities)
                    response = {
                        'status': 'success',
                        'intent': intent.primary_intent,
                        'confidence': intent.confidence,
                        'response': code_examples_dict.get('message', ''),
                        'message': code_examples_dict.get('message', ''),
                        'code_examples': code_examples_dict.get('code_examples', []),
                        'parameters': self._format_parameters(relevant_apis[0].parameters if relevant_apis else []),
                        'authentication': self._get_auth_info()
                    }
                else:
                    response = {
                        'status': 'success',
                        'intent': intent.primary_intent,
                        'confidence': 0.5,
                        'response': "I couldn't find a specific API for your example. Here's a general template:\n\n" + 
                                   "```python\nimport httpx\n\nheaders = {\n    'entity': 'GPRDTTSTOU',\n    'userId': '1'\n}\n\nresponse = httpx.get('https://api.example.com/endpoint', headers=headers)\n```",
                        'message': "Showing general API example"
                    }
            
            elif intent.primary_intent == 'find_api':
                if relevant_apis:
                    api_list = '\n'.join([
                        f"• **{api.method} {api.path}**\n  _{self.api_metadata.get(api.endpoint_name, {}).get('summary', 'No description')[:100]}_"
                        for api in relevant_apis[:5]
                    ])
                    
                    response = {
                        'status': 'success',
                        'intent': intent.primary_intent,
                        'confidence': intent.confidence,
                        'response': f"I found **{len(relevant_apis)} relevant APIs** for your query:\n\n{api_list}\n\nWould you like code examples for any of these?",
                        'message': f"Found {len(relevant_apis)} relevant APIs",
                        'related_apis': [
                            {
                                'name': api.endpoint_name,
                                'confidence': api.confidence,
                                'method': api.method,
                                'path': api.path,
                                'description': self.api_metadata.get(api.endpoint_name, {}).get('description', '')
                            } for api in relevant_apis[:5]
                        ]
                    }
                else:
                    response = {
                        'status': 'success',
                        'intent': intent.primary_intent,
                        'confidence': 0.3,
                        'response': "I couldn't find specific APIs matching your query. Try asking about:\n\n" +
                                   "• Customer management\n• Account operations\n• Transaction processing\n• Loan management",
                        'message': "No matching APIs found"
                    }
            
            elif intent.primary_intent == 'debug_error':
                error_info = entities.get('ERROR_CODE', ['unknown'])[0] if 'ERROR_CODE' in entities else 'unknown'
                error_analysis = await self._analyze_error(query, entities, context)
                
                response = {
                    'status': 'success',
                    'intent': intent.primary_intent,
                    'confidence': intent.confidence,
                    'response': f"Let me help you debug error **{error_info}**:\n\n" + 
                               f"**Error Type:** {error_analysis['error_analysis']['error_name']}\n\n" +
                               f"**Common Causes:**\n" + 
                               '\n'.join([f"• {cause}" for cause in error_analysis['error_analysis']['common_causes']]) + "\n\n" +
                               f"**Solutions:**\n" +
                               '\n'.join([f"• {solution}" for solution in error_analysis['error_analysis']['solutions']]),
                    'message': f"Debugging error {error_info}",
                    'error_analysis': error_analysis.get('error_analysis', {}),
                    'debugging_steps': error_analysis.get('debugging_steps', [])
                }
            
            elif intent.primary_intent == 'authentication':
                response = {
                    'status': 'success',
                    'intent': intent.primary_intent,
                    'confidence': intent.confidence,
                    'response': "**BaNCS Authentication Headers:**\n\n" +
                               "All API requests require these headers:\n\n" +
                               "```json\n{\n" +
                               '  "entity": "GPRDTTSTOU",\n' +
                               '  "userId": "1",\n' +
                               '  "languageCode": "1",\n' +
                               '  "Content-Type": "application/json"\n' +
                               "}\n```\n\n" +
                               "Make sure to include these in every request!",
                    'message': "Authentication information",
                    'authentication': self._get_auth_info()
                }
            
            else:
                # Default response
                response = {
                    'status': 'success',
                    'intent': intent.primary_intent,
                    'confidence': intent.confidence,
                    'response': "I can help you with API endpoints, code examples, and troubleshooting. What would you like to know?",
                    'message': "How can I help you with the APIs?",
                    'suggestions': [
                        "Ask for code examples for specific endpoints",
                        "Get help debugging API errors",
                        "Find relevant APIs for your use case"
                    ]
                }
            
            # Ensure we always have a 'response' field for the frontend
            if 'response' not in response:
                response['response'] = response.get('message', 'Processing complete')
            
            return response
            
        except Exception as e:
            logger.error(f"[ST] Error processing query: {e}", exc_info=True)
            return {
                'status': 'error',
                'response': "I encountered an error processing your query. Please try again.",
                'message': f"Error: {str(e)}",
                'intent': 'unknown',
                'confidence': 0.0
            }
    
    def _find_relevant_apis(
        self,
        query_embeddings,
        intent: QueryIntent,
        entities: Dict[str, List[str]]
    ) -> List[APIMatch]:
        """Find relevant APIs using TF-IDF similarity"""
        
        # Check for exact name matches first
        query_lower = self.current_query.lower() if hasattr(self, 'current_query') else ""
        exact_matches = []
        
        # Use knowledge base if available
        if self.kb and hasattr(self.kb, 'endpoints'):
            for endpoint_name, endpoint_data in self.kb.endpoints.items():
                endpoint_normalized = endpoint_name.lower().replace('_', '').replace('-', '').replace(' ', '')
                query_normalized = query_lower.replace('_', '').replace('-', '').replace(' ', '')
                
                # Check for keyword matches
                if any(word in endpoint_normalized for word in query_normalized.split()) or \
                   any(word in query_normalized for word in endpoint_normalized.split()):
                    exact_matches.append(APIMatch(
                        endpoint_name=f"{endpoint_data.get('method', 'GET')} {endpoint_data.get('url', '/')}",
                        method=endpoint_data.get('method', 'GET'),
                        path=endpoint_data.get('url', '/'),
                        confidence=0.8,
                        parameters=list(endpoint_data.get('parameters', {}).values()) if isinstance(endpoint_data.get('parameters'), dict) else []
                    ))
        
        if exact_matches:
            return exact_matches[:5]
        
        # Use TF-IDF similarity
        matches = []
        
        for endpoint_key, api_embedding in self.api_embeddings.items():
            # Calculate cosine similarity
            similarity = cosine_similarity(query_embeddings, api_embedding)[0][0]
            
            # Boost based on entity matches
            boost = 1.0
            metadata = self.api_metadata.get(endpoint_key, {})
            
            # Check HTTP method match
            if 'HTTP_METHOD' in entities:
                for method in entities['HTTP_METHOD']:
                    if method.upper() == metadata.get('method', ''):
                        boost *= 1.5
            
            # Check for keyword matches in query
            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 3:  # Skip short words
                    if word in metadata.get('path', '').lower():
                        boost *= 1.3
                    if word in metadata.get('description', '').lower():
                        boost *= 1.2
            
            final_score = similarity * boost
            
            if final_score > 0.1:  # Lower threshold for TF-IDF
                matches.append(APIMatch(
                    endpoint_name=endpoint_key,
                    method=metadata.get('method', 'GET'),
                    path=metadata.get('path', '/'),
                    confidence=final_score,
                    parameters=metadata.get('parameters', [])
                ))
        
        # Sort by confidence
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches[:5]
    
    def _generate_code_example(self, apis: List[APIMatch], entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Generate code examples"""
        if not apis:
            return {
                'message': "I'll show you a general example for API usage:",
                'code_examples': [self._create_generic_example()]
            }
        
        best_match = apis[0]
        examples = []
        message = f"Here's how to use the **{best_match.endpoint_name}** endpoint:\n\n"
        message += f"**Endpoint:** `{best_match.method} {best_match.path}`\n\n"
        
        # Generate Python example by default
        example = self._create_code_example(best_match, 'python')
        if example and example.get('code'):
            examples.append({
                'language': 'python',
                'code': example['code'],
                'description': example.get('description', 'Python example'),
                'notes': example.get('notes', [])
            })
        
        return {
            'message': message,
            'code_examples': examples,
            'parameters': self._format_parameters(best_match.parameters),
            'authentication': self._get_auth_info(),
            'rate_limits': self._get_rate_limit_info(best_match)
        }
    
    def _create_code_example(self, api: APIMatch, language: str) -> Dict[str, Any]:
        """Create code example using templates"""
        # Simplified version - just generate Python example
        func_name = api.endpoint_name.lower().replace(' ', '_').replace('-', '_').replace('/', '_')
        
        code = f"""import httpx

async def call_{func_name}():
    \"\"\"Call {api.endpoint_name} endpoint\"\"\"
    
    headers = {{
        'entity': 'GPRDTTSTOU',
        'userId': '1',
        'languageCode': '1',
        'Content-Type': 'application/json'
    }}
    
    url = "https://demoapps.tcsbancs.com{api.path}"
    
    async with httpx.AsyncClient() as client:
        response = await client.{api.method.lower()}(
            url,
            headers=headers
        )
        return response.json()

# Usage
import asyncio
result = asyncio.run(call_{func_name}())
print(result)
"""
        
        return {
            'code': code.strip(),
            'description': f'{language.title()} example for {api.endpoint_name}',
            'notes': [
                f'Method: {api.method}',
                f'Endpoint: {api.path}',
                'Remember to handle errors appropriately'
            ]
        }
    
    def _create_generic_example(self) -> Dict[str, Any]:
        """Create a generic API example"""
        code = """import httpx

headers = {
    'entity': 'GPRDTTSTOU',
    'userId': '1',
    'languageCode': '1',
    'Content-Type': 'application/json'
}

url = 'https://demoapps.tcsbancs.com/api/endpoint'

async with httpx.AsyncClient() as client:
    response = await client.get(url, headers=headers)
    
    if response.status_code == 200:
        print('Success:', response.json())
    else:
        print('Error:', response.status_code)
"""
        
        return {
            'language': 'python',
            'code': code,
            'description': 'Generic API example'
        }
    
    async def _analyze_error(self, query: str, entities: Dict[str, List[str]], context) -> Dict[str, Any]:
        """Analyze error"""
        error_code = entities.get('ERROR_CODE', ['unknown'])[0] if 'ERROR_CODE' in entities else 'unknown'
        
        error_solutions = {
            '401': {
                'name': 'Unauthorized',
                'common_causes': ['Missing authentication headers', 'Invalid credentials', 'Expired token'],
                'solutions': ['Check entity, userId, languageCode headers', 'Verify credentials are correct', 'Ensure headers are properly formatted']
            },
            '403': {
                'name': 'Forbidden',
                'common_causes': ['Insufficient permissions', 'Resource access denied'],
                'solutions': ['Check user permissions', 'Verify API access rights']
            },
            '404': {
                'name': 'Not Found',
                'common_causes': ['Invalid resource ID', 'Wrong endpoint path', 'Resource deleted'],
                'solutions': ['Verify resource exists', 'Check endpoint path spelling', 'Confirm API version']
            },
            '500': {
                'name': 'Internal Server Error',
                'common_causes': ['Server-side issue', 'Invalid request format'],
                'solutions': ['Retry with exponential backoff', 'Check request body format', 'Contact API support']
            }
        }
        
        error_info = error_solutions.get(error_code, {
            'name': 'Unknown Error',
            'common_causes': ['Error not recognized'],
            'solutions': ['Check API documentation', 'Verify request format']
        })
        
        return {
            'error_analysis': {
                'error_code': error_code,
                'error_name': error_info['name'],
                'common_causes': error_info['common_causes'],
                'solutions': error_info['solutions']
            },
            'debugging_steps': [
                'Check request headers format',
                'Verify authentication credentials',
                'Validate request parameters',
                'Review API documentation',
                'Test with minimal request'
            ],
            'code_snippet': f"# Debug headers\nprint('Headers:', headers)\n# Check response\nprint('Status:', response.status_code)\nprint('Body:', response.text)"
        }
    
    def _format_parameters(self, parameters: List[Dict]) -> List[Dict[str, Any]]:
        """Format parameters for response"""
        if not parameters:
            return []
        
        formatted = []
        for param in parameters[:10]:
            if isinstance(param, dict):
                formatted.append({
                    'name': param.get('name', 'unknown'),
                    'type': param.get('type', 'string'),
                    'required': param.get('required', False),
                    'description': param.get('description', ''),
                    'in': param.get('in', 'query')
                })
        
        return formatted
    
    def _get_auth_info(self) -> Dict[str, Any]:
        """Get authentication information"""
        return {
            'type': 'custom_headers',
            'headers': {
                'entity': 'GPRDTTSTOU',
                'userId': '1',
                'languageCode': '1',
                'Content-Type': 'application/json'
            },
            'description': 'BaNCS requires custom headers for authentication'
        }
    
    def _get_rate_limit_info(self, api: APIMatch) -> Dict[str, Any]:
        """Get rate limit information"""
        return {
            'requests_per_minute': 60,
            'requests_per_hour': 1000,
            'retry_after_header': True,
            'recommendation': 'Implement exponential backoff'
        }

# Alias for compatibility
BERTCookbookChatbot = SentenceTransformerChatbot