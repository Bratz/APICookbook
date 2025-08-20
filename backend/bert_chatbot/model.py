"""
BERT-based chatbot model for API Cookbook
Uses CodeBERT for understanding code-related queries
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    pipeline
)
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import json
from pathlib import Path
import logging
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class QueryIntent:
    """Represents the intent of a developer query"""
    primary_intent: str  # e.g., 'get_example', 'debug_error', 'find_api'
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
    
class BERTCookbookChatbot:
    """
    Advanced BERT-based chatbot for API documentation
    Specialized for developer queries and code understanding
    """
    
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        # Initialize CodeBERT for code understanding
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name)
        
        # Initialize specialized models
        self.intent_classifier = self._build_intent_classifier()
        self.entity_extractor = self._build_entity_extractor()
        self.code_generator = self._build_code_generator()
        self.error_analyzer = self._build_error_analyzer()
        
        # Load API knowledge base
        self.api_embeddings = {}
        self.api_metadata = {}
        self.load_api_knowledge()
        
        # Initialize conversation context
        self.conversation_history = []
        self.user_preferences = {
            'preferred_language': 'python',
            'experience_level': 'intermediate',
            'verbosity': 'normal'
        }
        
    def _build_intent_classifier(self) -> nn.Module:
        """Build intent classification model"""
        class IntentClassifier(nn.Module):
            def __init__(self, hidden_size=768, num_intents=15):
                super().__init__()
                self.dropout = nn.Dropout(0.1)
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_size, 256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, num_intents)
                )
                self.intent_labels = [
                    'get_example',        # User wants code example
                    'find_api',          # Looking for specific API
                    'debug_error',       # Has an error to debug
                    'understand_params', # Needs parameter explanation
                    'compare_apis',      # Comparing multiple APIs
                    'best_practices',    # Asking for best practices
                    'authentication',    # Auth-related query
                    'rate_limiting',     # Rate limit questions
                    'pagination',        # Pagination help
                    'filtering',         # Data filtering
                    'create_resource',   # POST/PUT operations
                    'update_resource',   # PATCH operations
                    'delete_resource',   # DELETE operations
                    'webhook_setup',     # Webhook configuration
                    'testing'           # Testing/mocking APIs
                ]
                
            def forward(self, embeddings):
                return self.classifier(self.dropout(embeddings))
        
        return IntentClassifier()
    
    def _build_entity_extractor(self) -> nn.Module:
        """Build entity extraction model for API-specific entities"""
        class EntityExtractor(nn.Module):
            def __init__(self, hidden_size=768):
                super().__init__()
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
                self.lstm = nn.LSTM(hidden_size, 256, bidirectional=True, batch_first=True)
                self.classifier = nn.Linear(512, len(self.entity_types) + 1)  # +1 for 'O' (outside)
                
            def forward(self, embeddings):
                lstm_out, _ = self.lstm(embeddings)
                return self.classifier(lstm_out)
        
        return EntityExtractor()
    
    def _build_code_generator(self):
        """Build code generation model"""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            
            class CodeGenerator:
                def __init__(self):
                    try:
                        # Use Auto classes to get the correct tokenizer/model pair
                        model_name = "Salesforce/codet5-small"  # Use smaller model
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                        self.available = True
                    except Exception as e:
                        logger.warning(f"Could not load code generation model: {e}")
                        self.available = False
                        
                def generate(self, prompt: str, max_length: int = 256) -> str:
                    if not self.available:
                        return "# Code generation not available"
                    
                    try:
                        inputs = self.tokenizer(
                            prompt,
                            return_tensors="pt",
                            max_length=512,
                            truncation=True
                        )
                        
                        outputs = self.model.generate(
                            inputs.input_ids,
                            max_length=max_length,
                            num_return_sequences=1,
                            temperature=0.7,
                            do_sample=True
                        )
                        
                        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    except Exception as e:
                        logger.error(f"Code generation error: {e}")
                        return "# Error generating code"
            
            return CodeGenerator()
        
        except ImportError:
            # If transformers not available, return dummy generator
            class DummyGenerator:
                def __init__(self):
                    self.available = False
                
                def generate(self, prompt: str, max_length: int = 256) -> str:
                    return "# Code generation requires transformers library"
            
            return DummyGenerator()
        
    def _build_error_analyzer(self) -> nn.Module:
        """Build error analysis model"""
        class ErrorAnalyzer(nn.Module):
            def __init__(self, hidden_size=768):
                super().__init__()
                self.error_patterns = nn.Sequential(
                    nn.Linear(hidden_size, 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128)  # Error pattern embeddings
                )
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
                
            def forward(self, embeddings):
                return self.error_patterns(embeddings)
        
        return ErrorAnalyzer()
    
    def load_api_knowledge(self):
        """Load and index API documentation"""
        # Load API specifications
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
        """Index API specification for semantic search with enriched context"""
        # Extract and encode API endpoints
        for path, methods in spec.get('paths', {}).items():
            for method, details in methods.items():
                if method in ['get', 'post', 'put', 'delete', 'patch']:
                    endpoint_key = f"{method.upper()} {path}"
                    
                    # Build comprehensive description with multiple context sources
                    description_parts = []
                    
                    # 1. Basic description
                    summary = details.get('summary', '')
                    desc = details.get('description', '')
                    if summary:
                        description_parts.append(summary)
                    if desc:
                        description_parts.append(desc)
                    
                    # 2. Extract operation context from path
                    path_context = self._extract_path_context(path)
                    if path_context:
                        description_parts.append(f"Related to: {path_context}")
                    
                    # 3. Add parameter information for better matching
                    parameters = details.get('parameters', [])
                    if parameters:
                        param_names = []
                        param_descriptions = []
                        for param in parameters:
                            if isinstance(param, dict):
                                name = param.get('name', '')
                                if name:
                                    param_names.append(name)
                                param_desc = param.get('description', '')
                                if param_desc:
                                    param_descriptions.append(param_desc)
                        
                        if param_names:
                            description_parts.append(f"Parameters: {', '.join(param_names)}")
                        if param_descriptions:
                            description_parts.append(' '.join(param_descriptions))
                    
                    # 4. Add request body context
                    request_body = details.get('requestBody', {})
                    if request_body:
                        body_desc = request_body.get('description', '')
                        if body_desc:
                            description_parts.append(f"Request: {body_desc}")
                        
                        # Extract schema properties if available
                        content = request_body.get('content', {})
                        if 'application/json' in content:
                            schema = content['application/json'].get('schema', {})
                            properties = self._extract_schema_properties(schema)
                            if properties:
                                description_parts.append(f"Input fields: {', '.join(properties)}")
                    
                    # 5. Add response context - MOST IMPORTANT for queries like "get mobile number"
                    responses = details.get('responses', {})
                    response_fields = []
                    for status_code, response_data in responses.items():
                        if status_code.startswith('2'):  # Success responses
                            resp_desc = response_data.get('description', '')
                            if resp_desc:
                                description_parts.append(f"Returns: {resp_desc}")
                            
                            # Extract response schema
                            content = response_data.get('content', {})
                            if 'application/json' in content:
                                schema = content['application/json'].get('schema', {})
                                fields = self._extract_schema_properties(schema)
                                response_fields.extend(fields)
                    
                    if response_fields:
                        description_parts.append(f"Response includes: {', '.join(set(response_fields))}")
                    
                    # 6. Add tags for categorical context
                    tags = details.get('tags', [])
                    if tags:
                        description_parts.append(f"Categories: {', '.join(tags)}")
                    
                    # 7. Add operation ID as it often contains meaningful names
                    operation_id = details.get('operationId', '')
                    if operation_id:
                        # Convert camelCase to space-separated words
                        readable_op = self._camel_to_readable(operation_id)
                        description_parts.append(f"Operation: {readable_op}")
                    
                    # Combine all parts into comprehensive description
                    full_description = ' | '.join(filter(None, description_parts))
                    
                    # Generate embeddings from the enriched description
                    embeddings = self._generate_embeddings(full_description)
                    
                    # Store both embeddings and metadata
                    self.api_embeddings[endpoint_key] = embeddings
                    self.api_metadata[endpoint_key] = {
                        'method': method.upper(),
                        'path': path,
                        'parameters': parameters,
                        'request_body': request_body,
                        'responses': responses,
                        'tags': tags,
                        'operation_id': operation_id,
                        'description': desc,  # Store original description
                        'summary': summary,  # Store original summary
                        'full_context': full_description  # Store what we used for embeddings
                    }
                    
                    logger.debug(f"Indexed {endpoint_key} with context: {full_description[:200]}...")

    def _extract_path_context(self, path: str) -> str:
        """Extract meaningful context from API path"""
        # Remove parameter placeholders and extract meaningful parts
        parts = []
        for segment in path.split('/'):
            if segment and not segment.startswith('{'):
                # Convert to readable form
                readable = segment.replace('_', ' ').replace('-', ' ')
                parts.append(readable)
        
        return ' '.join(parts)

    def _extract_schema_properties(self, schema: Dict, max_depth: int = 2, current_depth: int = 0) -> List[str]:
        """Recursively extract property names from schema"""
        if current_depth >= max_depth:
            return []
        
        properties = []
        
        # Direct properties
        if 'properties' in schema:
            for prop_name, prop_schema in schema.get('properties', {}).items():
                properties.append(prop_name)
                # Look for nested properties
                if isinstance(prop_schema, dict):
                    nested = self._extract_schema_properties(prop_schema, max_depth, current_depth + 1)
                    properties.extend([f"{prop_name}.{n}" for n in nested if n])
        
        # Array items
        if schema.get('type') == 'array' and 'items' in schema:
            items_props = self._extract_schema_properties(schema['items'], max_depth, current_depth + 1)
            properties.extend(items_props)
        
        # OneOf, AnyOf, AllOf schemas
        for key in ['oneOf', 'anyOf', 'allOf']:
            if key in schema:
                for sub_schema in schema[key]:
                    properties.extend(self._extract_schema_properties(sub_schema, max_depth, current_depth + 1))
        
        # Common fields that might indicate what data is returned
        field_hints = {
            'mobile': ['mobile', 'mobileNumber', 'mobilePhone', 'phone', 'phoneNumber', 'contactNumber'],
            'email': ['email', 'emailAddress', 'emailId'],
            'address': ['address', 'streetAddress', 'city', 'state', 'zipCode', 'postalCode'],
            'name': ['name', 'firstName', 'lastName', 'fullName', 'customerName'],
            'balance': ['balance', 'accountBalance', 'availableBalance', 'currentBalance'],
            'account': ['accountNumber', 'accountId', 'accountRef'],
            'customer': ['customerId', 'customerRef', 'cifNumber']
        }
        
        # Check if properties match common patterns
        for category, patterns in field_hints.items():
            for prop in properties:
                prop_lower = prop.lower()
                if any(pattern.lower() in prop_lower for pattern in patterns):
                    properties.append(f"[{category}_field]")  # Add a hint
        
        return list(set(properties))  # Remove duplicates

    def _camel_to_readable(self, camel_case: str) -> str:
        """Convert camelCase or PascalCase to readable text"""
        import re
        # Insert spaces before capital letters
        spaced = re.sub(r'(?<!^)(?=[A-Z])', ' ', camel_case)
        return spaced.lower()
    
    def _generate_embeddings(self, text: str) -> torch.Tensor:
        """Generate BERT embeddings for text"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.base_model(**inputs)
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings.squeeze()
    
    async def process_query(
        self, 
        query: str,
        context: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Process a developer query and generate response"""
        try:
            logger.info(f"[BERT] Query: '{query}'")
            self.current_query = query
            
            # Tokenize and encode the query to get embeddings
            inputs = self.tokenizer(
                query,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # Get query embeddings from the model
            with torch.no_grad():
                outputs = self.base_model(**inputs)
                # Use the [CLS] token embedding as the query representation
                query_embeddings = outputs.last_hidden_state[:, 0, :].squeeze()
            
            # Classify intent - FIX: pass both parameters
            intent_result = self._classify_intent(query_embeddings, query)
            intent = intent_result.primary_intent
            confidence = intent_result.confidence
            entities = intent_result.entities
            
            logger.info(f"[BERT] Classified intent: {intent} (confidence: {confidence:.2f})")
            
            # Find relevant APIs with the embeddings
            relevant_apis = self._find_relevant_apis(query_embeddings, intent_result, entities)
            
            logger.info(f"[BERT] Found {len(relevant_apis)} relevant APIs")
            
            if relevant_apis:
                top_match = relevant_apis[0]
                logger.info(f"[BERT] Top match: {top_match.endpoint_name} (confidence: {top_match.confidence:.2f})")
            
            # Generate response based on intent
            response = {}
            
            if intent == 'get_example':
                if relevant_apis:
                    # FIX: Call with correct parameters
                    code_examples_dict = self._generate_code_example(relevant_apis, entities)
                    code_examples = code_examples_dict.get('code_examples', [])
                    
                    response = {
                        'status': 'success',
                        'intent': intent,
                        'confidence': confidence,
                        'message': code_examples_dict.get('message', f"Here's how to use the **{relevant_apis[0].endpoint_name}** endpoint:"),
                        'code_examples': code_examples,
                        'api_references': [],
                        'related_apis': [
                            {
                                'name': api.endpoint_name,
                                'confidence': api.confidence,
                                'method': api.method,
                                'path': api.path
                            } for api in relevant_apis[:3]
                        ]
                    }
                    
                    # Add parameters if available
                    if relevant_apis[0].parameters:
                        response['parameters'] = self._format_parameters(relevant_apis[0].parameters)
                    
                    # Add authentication info
                    response['authentication'] = self._get_auth_info()
                    
                    # Add suggestions
                    response['suggestions'] = [
                        "Try the example with your API credentials",
                        "Modify parameters to suit your use case", 
                        "Add error handling for production use"
                    ]
                    
            elif intent == 'debug_error':
                # FIX: Extract error info properly and use await
                error_info = entities.get('ERROR_CODE', ['unknown'])[0] if 'ERROR_CODE' in entities else 'unknown'
                error_analysis = await self._analyze_error(query, entities, context)
                
                response = {
                    'status': 'success',
                    'intent': intent,
                    'confidence': confidence,
                    'message': f"Let me help you debug the error: {error_info}",
                    'error_analysis': error_analysis.get('error_analysis', {}),
                    'debugging_steps': error_analysis.get('debugging_steps', []),
                    'code_snippet': error_analysis.get('code_snippet', ''),
                    'suggestions': [
                        "Check your authentication headers",
                        "Verify the API endpoint URL",
                        "Ensure all required parameters are provided",
                        "Check the API rate limits"
                    ]
                }
                
            elif intent == 'find_api':
                response = {
                    'status': 'success',
                    'intent': intent,
                    'confidence': confidence,
                    'message': f"I found {len(relevant_apis)} relevant APIs for your query:",
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
                # Default response
                response = {
                    'status': 'success',
                    'intent': intent,
                    'confidence': confidence,
                    'message': "I can help you with API endpoints, code examples, and troubleshooting. What would you like to know?",
                    'suggestions': [
                        "Ask for code examples for specific endpoints",
                        "Get help debugging API errors",
                        "Find relevant APIs for your use case"
                    ]
                }
            
            return response
            
        except Exception as e:
            logger.error(f"[BERT] Error processing query: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': f"Error processing query: {str(e)}",
                'intent': 'unknown',
                'confidence': 0.0
            }
    
    def _classify_intent(self, embeddings: torch.Tensor, query: str = None) -> QueryIntent:
        """Classify the intent of the query with strong rule-based override for code requests"""
        
        # Default to neural network classification
        with torch.no_grad():
            logits = self.intent_classifier(embeddings)
            probs = torch.softmax(logits, dim=-1)
            nn_confidence, predicted = torch.max(probs, dim=-1)
            
            nn_intent = self.intent_classifier.intent_labels[predicted.item()]
            nn_conf = nn_confidence.item()
        
        # ALWAYS use rule-based classification when we have the query text
        if query:
            query_lower = query.lower()
            
            # Strong patterns for code examples - CHECK THESE FIRST
            code_keywords = ['code', 'example', 'sample', 'snippet', 'show me', 'give me', 'how to', 'python', 'javascript', 'java', 'curl', 'c#', 'ruby', 'php']
            code_score = sum(1 for keyword in code_keywords if keyword in query_lower)
            
            # Strong patterns for each intent
            if any(word in query_lower for word in ['code', 'example', 'sample', 'snippet', 'with examples']):
                # Definitely asking for code
                return QueryIntent(
                    primary_intent='get_example',
                    confidence=0.95,
                    entities={},
                    context_required=False
                )
            
            if 'how to' in query_lower or 'how do i' in query_lower:
                # Asking how to do something - usually wants code
                if any(word in query_lower for word in ['python', 'javascript', 'java', 'curl', 'code']):
                    return QueryIntent(
                        primary_intent='get_example',
                        confidence=0.9,
                        entities={},
                        context_required=False
                    )
                # Even without language specified, "how to" usually wants examples
                return QueryIntent(
                    primary_intent='get_example',
                    confidence=0.8,
                    entities={},
                    context_required=False
                )
            
            if 'show' in query_lower or 'give' in query_lower or 'tell me about' in query_lower:
                # Asking to show/give something
                if any(word in query_lower for word in ['endpoint', 'api', 'example']):
                    return QueryIntent(
                        primary_intent='get_example',
                        confidence=0.9,
                        entities={},
                        context_required=False
                    )
            
            # Check for error-related queries
            error_keywords = ['error', 'fail', 'wrong', 'issue', 'problem', 'bug', '400', '401', '403', '404', '500', 'exception', "doesn't work", "not working"]
            if any(keyword in query_lower for keyword in error_keywords):
                return QueryIntent(
                    primary_intent='debug_error',
                    confidence=0.85,
                    entities={},
                    context_required=True
                )
            
            # Check for authentication queries
            auth_keywords = ['auth', 'authenticate', 'header', 'token', 'credential', 'login', 'security', 'entity', 'userid', 'languagecode']
            if any(keyword in query_lower for keyword in auth_keywords):
                return QueryIntent(
                    primary_intent='authentication',
                    confidence=0.85,
                    entities={},
                    context_required=False
                )
            
            # Check for API finding queries
            find_keywords = ['which api', 'what api', 'find', 'search', 'endpoint for', 'api for', 'list', 'available']
            if any(keyword in query_lower for keyword in find_keywords):
                return QueryIntent(
                    primary_intent='find_api',
                    confidence=0.8,
                    entities={},
                    context_required=False
                )
            
            # Check for parameter queries
            param_keywords = ['parameter', 'param', 'argument', 'input', 'require', 'optional', 'mandatory', 'field']
            if any(keyword in query_lower for keyword in param_keywords):
                return QueryIntent(
                    primary_intent='understand_params',
                    confidence=0.8,
                    entities={},
                    context_required=False
                )
            
            # Check for best practices
            if 'best practice' in query_lower or 'recommendation' in query_lower or 'guide' in query_lower:
                return QueryIntent(
                    primary_intent='best_practices',
                    confidence=0.8,
                    entities={},
                    context_required=False
                )
            
            # Check for CRUD operations
            if any(word in query_lower for word in ['create', 'add', 'new', 'insert', 'post']):
                # Could be asking for code to create something
                if code_score > 0:
                    return QueryIntent(
                        primary_intent='get_example',
                        confidence=0.85,
                        entities={},
                        context_required=False
                    )
                return QueryIntent(
                    primary_intent='create_resource',
                    confidence=0.7,
                    entities={},
                    context_required=False
                )
            
            if any(word in query_lower for word in ['update', 'modify', 'change', 'edit', 'patch']):
                return QueryIntent(
                    primary_intent='update_resource',
                    confidence=0.7,
                    entities={},
                    context_required=False
                )
            
            if any(word in query_lower for word in ['delete', 'remove', 'cancel']):
                return QueryIntent(
                    primary_intent='delete_resource',
                    confidence=0.7,
                    entities={},
                    context_required=False
                )
            
            # If nothing matched strongly, but there are code-related words, assume they want examples
            if code_score >= 2:
                return QueryIntent(
                    primary_intent='get_example',
                    confidence=0.7,
                    entities={},
                    context_required=False
                )
        
        # Fall back to neural network if no strong patterns matched
        # But boost confidence slightly since our NN isn't trained
        return QueryIntent(
            primary_intent=nn_intent,
            confidence=min(nn_conf * 1.2, 0.5),  # Boost but cap at 0.5
            entities={},
            context_required=nn_intent in ['debug_error', 'compare_apis']
        )
    
    def _extract_entities(self, query: str, embeddings: torch.Tensor) -> Dict[str, List[str]]:
        """Extract API-specific entities from query"""
        tokens = self.tokenizer.tokenize(query)
        
        # Get token-level predictions
        with torch.no_grad():
            # Expand embeddings to match token length
            token_embeddings = embeddings.unsqueeze(0).repeat(1, len(tokens), 1)
            predictions = self.entity_extractor(token_embeddings)
            predicted_labels = torch.argmax(predictions, dim=-1)
        
        # Group entities by type
        entities = {}
        current_entity = []
        current_type = None
        
        for token, label_id in zip(tokens, predicted_labels[0]):
            if label_id < len(self.entity_extractor.entity_types):
                entity_type = self.entity_extractor.entity_types[label_id]
                
                if entity_type != current_type:
                    if current_entity and current_type:
                        if current_type not in entities:
                            entities[current_type] = []
                        entities[current_type].append(' '.join(current_entity))
                    current_entity = [token.replace('##', '')]
                    current_type = entity_type
                else:
                    current_entity.append(token.replace('##', ''))
        
        # Add last entity
        if current_entity and current_type:
            if current_type not in entities:
                entities[current_type] = []
            entities[current_type].append(' '.join(current_entity))
        
        return entities
    
    def _find_relevant_apis(
        self, 
        query_embeddings: torch.Tensor, 
        intent: QueryIntent,
        entities: Dict[str, List[str]]
    ) -> List[APIMatch]:
        """Find relevant APIs using semantic search with exact name matching priority"""
        
        # First, check if the query mentions a specific endpoint name
        query_lower = self.current_query.lower() if hasattr(self, 'current_query') else ""
        exact_matches = []
        
        # Improved matching logic
        for endpoint_key in self.api_metadata.keys():
            # Normalize endpoint name for matching
            endpoint_name_normalized = endpoint_key.lower().replace('_', '').replace('-', '').replace(' ', '')
            query_normalized = query_lower.replace('_', '').replace('-', '').replace(' ', '')
            
            # Check for exact endpoint name in query
            if endpoint_name_normalized in query_normalized:
                metadata = self.api_metadata.get(endpoint_key, {})
                parameters = metadata.get('parameters', [])
                
                # Convert parameters to list format if needed
                if isinstance(parameters, dict):
                    parameters_list = []
                    for param_name, param_info in parameters.items():
                        param_dict = {'name': param_name}
                        if isinstance(param_info, dict):
                            param_dict.update(param_info)
                        parameters_list.append(param_dict)
                    parameters = parameters_list
                
                exact_matches.append(APIMatch(
                    endpoint_name=endpoint_key,
                    method=metadata.get('method', 'GET'),
                    path=metadata.get('path', '/'),
                    confidence=1.0,  # Perfect score for exact match
                    parameters=parameters if isinstance(parameters, list) else []
                ))
                logger.info(f"[BERT] Found exact endpoint match: {endpoint_key}")
        
        # If we found exact matches, return them
        if exact_matches:
            return exact_matches[:5]
        
        # Otherwise, use semantic similarity
        matches = []
        
        # Calculate cosine similarity with all API endpoints
        for endpoint_key, api_embeddings in self.api_embeddings.items():
            similarity = torch.cosine_similarity(query_embeddings, api_embeddings, dim=0)
            
            # Boost score based on entity matches
            boost = 1.0
            metadata = self.api_metadata.get(endpoint_key, {})
            
            # Check HTTP method match
            if 'HTTP_METHOD' in entities:
                for method in entities['HTTP_METHOD']:
                    if method.upper() == metadata.get('method', ''):
                        boost *= 1.5
            
            # Check parameter matches
            if 'PARAMETER_NAME' in entities:
                parameters = metadata.get('parameters', [])
                
                # Handle both list and dict formats
                param_names = []
                if isinstance(parameters, list):
                    param_names = [p.get('name', '') for p in parameters if isinstance(p, dict)]
                elif isinstance(parameters, dict):
                    param_names = list(parameters.keys())
                
                for entity_param in entities['PARAMETER_NAME']:
                    if any(entity_param.lower() in param.lower() for param in param_names):
                        boost *= 1.3
            
            final_score = similarity.item() * boost
            
            if final_score > 0.5:  # Threshold
                parameters = metadata.get('parameters', [])
                if isinstance(parameters, dict):
                    parameters_list = []
                    for param_name, param_info in parameters.items():
                        param_dict = {'name': param_name}
                        if isinstance(param_info, dict):
                            param_dict.update(param_info)
                        parameters_list.append(param_dict)
                    parameters = parameters_list
                
                matches.append(APIMatch(
                    endpoint_name=endpoint_key,
                    method=metadata.get('method', 'GET'),
                    path=metadata.get('path', '/'),
                    confidence=final_score,
                    parameters=parameters if isinstance(parameters, list) else []
                ))
        
        # Sort by confidence
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches[:5]
    
    def _generate_code_example(self, apis: List[APIMatch], entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Generate code examples for matched APIs"""
        if not apis:
            # If no APIs matched, provide a generic example
            return {
                'message': "I'll show you a general example for API usage:",
                'code_examples': [self._create_generic_example()]
            }
        
        best_match = apis[0]
        language = self._detect_language_preference(entities)
        
        # Log for debugging
        logger.info(f"Generating code for {best_match.endpoint_name}, language preference: {language}")
        
        examples = []
        message = f"Here's how to use the **{best_match.endpoint_name}** endpoint:\n\n"
        message += f"**Endpoint:** {best_match.method} {best_match.path}\n"
        
        # Generate examples for requested language or all if not specified
        languages_to_generate = []
        if language in ['python', 'javascript', 'curl', 'java']:
            languages_to_generate = [language]
        else:
            # Default to Python and cURL
            languages_to_generate = ['python', 'curl']
        
        for lang in languages_to_generate:
            try:
                example = self._create_code_example(best_match, lang)
                if example and example.get('code'):
                    examples.append({
                        'language': lang,
                        'code': example['code'],
                        'description': example.get('description', f'{lang.title()} example'),
                        'notes': example.get('notes', [])
                    })
                    logger.info(f"Generated {lang} example with {len(example["code"])} chars")
            except Exception as e:
                logger.error(f"Failed to generate {lang} example: {e}")
        
        # If no examples were generated, create a basic one
        if not examples:
            logger.warning("No examples generated, creating fallback")
            examples = [self._create_fallback_example(best_match)]
        
        return {
            'message': message,
            'code_examples': examples,
            'parameters': self._format_parameters(best_match.parameters),
            'authentication': self._get_auth_info(),
            'rate_limits': self._get_rate_limit_info(best_match)
        }

    def _create_fallback_example(self, api: APIMatch) -> Dict[str, Any]:
        """Create a fallback code example when generation fails"""
        code = f"""# {api.endpoint_name}
# Method: {api.method}
# Path: {api.path}

import requests

headers = {{
    'entity': 'GPRDTTSTOU',
    'userId': '1',
    'languageCode': '1',
    'Content-Type': 'application/json'
}}

url = 'https://demoapps.tcsbancs.com{api.path}'

response = requests.{api.method.lower()}(url, headers=headers)
print(response.json())
"""
        
        return {
            'language': 'python',
            'code': code,
            'description': 'Basic example'
        }

    def _create_generic_example(self) -> Dict[str, Any]:
        """Create a generic API example"""
        code = """# Generic BAnCS API Example

import requests

# Required headers for all BAnCS API calls
headers = {
    'entity': 'GPRDTTSTOU',
    'userId': '1',
    'languageCode': '1',
    'Content-Type': 'application/json'
}

# Example: Get Account Balance
account_ref = '101000000101814'
url = f'https://demoapps.tcsbancs.com/Core/accountManagement/account/balanceDetails/{account_ref}'

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    print('Account Balance:', data)
else:
    print(f'Error: {response.status_code}')
    print(response.text)
"""
        
        return {
            'language': 'python',
            'code': code,
            'description': 'Generic BAnCS API example'
        }

    def _create_code_example(self, api: APIMatch, language: str) -> Dict[str, Any]:
        """Create code example for specific language"""
        
        # Ensure parameters is a list
        if not isinstance(api.parameters, list):
            api.parameters = []
        
        # Filter out any non-dict parameters
        api.parameters = [p for p in api.parameters if isinstance(p, dict)]
        
        # Build path with example values
        example_path = api.path
        path_params = {}
        query_params = {}
        body_params = {}
        
        for param in api.parameters:
            param_name = param.get('name', '')
            param_in = param.get('in', 'query')
            
            # Generate example value
            if param.get('example'):
                value = param['example']
            elif param.get('type') == 'integer':
                value = 123
            elif param.get('type') == 'boolean':
                value = True
            else:
                value = f'example_{param_name}'
            
            if param_in == 'path':
                path_params[param_name] = value
                example_path = example_path.replace(f'{{{param_name}}}', str(value))
            elif param_in == 'query':
                query_params[param_name] = value
            elif param_in == 'body':
                body_params[param_name] = value
        
        if language == 'python':
            code = f"""import httpx
from typing import Dict, Any

async def call_{api.endpoint_name.lower().replace(' ', '_').replace('-', '_')}({self._generate_python_params(api.parameters)}):
    \"\"\"
    {api.endpoint_name}: {api.method} {api.path}
    \"\"\"
    
    client = httpx.AsyncClient()
    
    headers = {{
        'entity': 'GPRDTTSTOU',
        'userId': '1',
        'languageCode': '1',
        'Content-Type': 'application/json'
    }}
    """
            
            # Add query parameters if any
            if query_params:
                code += f"""
    params = {json.dumps(query_params, indent=4)}
    """
            else:
                code += """
    params = {}
    """
            
            # Add body if needed
            if body_params:
                code += f"""
    request_body = {json.dumps(body_params, indent=4)}
    """
            
            # Make the request
            if api.method == 'GET':
                code += f"""
    response = await client.get(
        url=f"https://demoapps.tcsbancs.com{example_path}",
        headers=headers,
        params=params
    )"""
            elif api.method in ['POST', 'PUT', 'PATCH']:
                if body_params:
                    code += f"""
    response = await client.{api.method.lower()}(
        url=f"https://demoapps.tcsbancs.com{example_path}",
        headers=headers,
        json=request_body
    )"""
                else:
                    code += f"""
    response = await client.{api.method.lower()}(
        url=f"https://demoapps.tcsbancs.com{example_path}",
        headers=headers
    )"""
            elif api.method == 'DELETE':
                code += f"""
    response = await client.delete(
        url=f"https://demoapps.tcsbancs.com{example_path}",
        headers=headers
    )"""
            
            code += """
    
    response.raise_for_status()
    return response.json()

# Example usage
async def main():
    result = await call_""" + api.endpoint_name.lower().replace(' ', '_').replace('-', '_') + """("""
            
            # Add example parameters
            example_args = []
            for param in api.parameters:
                if param.get('required'):
                    param_name = param.get('name', '')
                    if param.get('type') == 'integer':
                        example_args.append(f"{param_name}=123")
                    elif param.get('type') == 'boolean':
                        example_args.append(f"{param_name}=True")
                    else:
                        example_args.append(f'{param_name}="example_value"')
            
            code += ', '.join(example_args[:3])  # Limit to 3 for readability
            code += """)
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
"""
                
        elif language == 'javascript':
            code = f"""// {api.endpoint_name}
// Method: {api.method}
// Path: {api.path}

const axios = require('axios');

async function call{self._to_camel_case(api.endpoint_name)}("""
            
            # Add parameters
            js_params = []
            for param in api.parameters:
                if param.get('required'):
                    js_params.append(param.get('name', 'param'))
            code += ', '.join(js_params[:3])  # Limit parameters
            
            code += """) {
    // Required headers for BAnCS API
    const headers = {
        'entity': 'GPRDTTSTOU',
        'userId': '1',
        'languageCode': '1',
        'Content-Type': 'application/json'
    };
    
    // Build the URL
    const baseUrl = 'https://demoapps.tcsbancs.com';
    const url = `${baseUrl}""" + example_path + """`;
    """
            
            if query_params:
                code += f"""
    // Query parameters
    const params = {json.dumps(query_params, indent=4)};
    """
            
            if body_params:
                code += f"""
    // Request body
    const data = {json.dumps(body_params, indent=4)};
    """
            
            code += """
    try {
        // Make the API call
        const response = await axios({
            method: '""" + api.method + """',
            url: url,
            headers: headers,"""
            
            if query_params:
                code += """
            params: params,"""
            if body_params:
                code += """
            data: data,"""
            
            code += """
        });
        
        return response.data;
    } catch (error) {
        console.error('API call failed:', error.response?.data || error.message);
        throw error;
    }
}

// Example usage
(async () => {
    try {
        const result = await call""" + self._to_camel_case(api.endpoint_name) + """("""
            
            # Add example arguments
            js_args = []
            for param in api.parameters:
                if param.get('required'):
                    if param.get('type') == 'integer':
                        js_args.append("123")
                    elif param.get('type') == 'boolean':
                        js_args.append("true")
                    else:
                        js_args.append('"example_value"')
            code += ', '.join(js_args[:3])
            
            code += """);
        console.log('Success:', result);
    } catch (error) {
        console.error('Failed:', error);
    }
})();
"""
                
        elif language == 'curl':
            code = f"""#!/bin/bash

# {api.endpoint_name}
# Method: {api.method}
# Path: {api.path}

curl -X {api.method} \\
  "https://demoapps.tcsbancs.com{example_path}"""
            
            if query_params:
                query_string = '&'.join([f"{k}={v}" for k, v in query_params.items()])
                code += f"?{query_string}"
            
            code += """" \\
  -H "entity: GPRDTTSTOU" \\
  -H "userId: 1" \\
  -H "languageCode: 1" \\
  -H "Content-Type: application/json" """
            
            if body_params:
                code += f"""\\
  -d '{json.dumps(body_params, indent=2)}'"""
            
            code += """

# Pretty print with jq (if installed)
# Add: | jq '.' to the end of the curl command

# Save response to file
# Add: -o response.json

# Include response headers
# Add: -i
"""
                
        else:  # Java
            code = self._generate_java_code(api, example_path, query_params, body_params)
        
        return {
            'code': code.strip(),
            'description': f'{language.title()} example for {api.endpoint_name}',
            'notes': [
                f'Method: {api.method}',
                f'Endpoint: {api.path}',
                'Remember to handle errors appropriately',
                'Add retry logic for production use'
            ]
        }

    def _generate_python_params(self, parameters: List[Dict]) -> str:
        """Generate Python function parameters"""
        if not parameters:
            return ""
        
        params = []
        for param in parameters:
            if param.get('required'):
                param_name = param.get('name', 'param')
                params.append(param_name)
        
        return ', '.join(params)

    def _generate_java_code(self, api: APIMatch, path: str, query_params: Dict, body_params: Dict) -> str:
        """Generate Java code example"""
        return f"""import java.io.*;
import java.net.http.*;
import java.net.URI;
import java.util.HashMap;
import java.util.Map;
import com.fasterxml.jackson.databind.ObjectMapper;

public class {self._to_pascal_case(api.endpoint_name)}Example {{
    
    private static final String BASE_URL = "https://demoapps.tcsbancs.com";
    private static final HttpClient client = HttpClient.newHttpClient();
    private static final ObjectMapper mapper = new ObjectMapper();
    
    public static void main(String[] args) throws Exception {{
        // Build request
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(BASE_URL + "{path}"))
            .header("entity", "GPRDTTSTOU")
            .header("userId", "1")
            .header("languageCode", "1")
            .header("Content-Type", "application/json")
            .{api.method}(HttpRequest.BodyPublishers.ofString(""))
            .build();
        
        // Send request
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
        
        // Handle response
        if (response.statusCode() == 200) {{
            System.out.println("Success: " + response.body());
        }} else {{
            System.out.println("Error: " + response.statusCode());
        }}
    }}
}}"""
    
    async def _analyze_error(self, query: str, entities: Dict[str, List[str]], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error and provide debugging help"""
        
        # Extract error information
        error_code = entities.get('ERROR_CODE', ['unknown'])[0] if 'ERROR_CODE' in entities else 'unknown'
        
        # Common error patterns and solutions
        error_solutions = {
            '400': {
                'name': 'Bad Request',
                'common_causes': [
                    'Missing required parameters',
                    'Invalid parameter format',
                    'Malformed JSON in request body'
                ],
                'solutions': [
                    'Verify all required parameters are included',
                    'Check parameter data types match API specification',
                    'Validate JSON syntax using a JSON validator',
                    'Ensure date formats are correct (YYYYMMDD for BAnCS)'
                ],
                'code_snippet': """
# Debugging 400 errors
import json

# Log the request for debugging
print("Request Headers:", headers)
print("Request Body:", json.dumps(request_body, indent=2))

# Validate required fields
required_fields = ['customerId', 'accountType', 'currency']
missing_fields = [field for field in required_fields if field not in request_body]
if missing_fields:
    print(f"Missing required fields: {missing_fields}")
"""
            },
            '401': {
                'name': 'Unauthorized',
                'common_causes': [
                    'Missing authentication headers',
                    'Invalid credentials',
                    'Expired session token'
                ],
                'solutions': [
                    'Ensure entity, userId, and languageCode headers are present',
                    'Verify credentials are correct',
                    'Check if session has expired and needs refresh'
                ],
                'code_snippet': """
# Authentication headers required for BAnCS
headers = {
    'entity': 'GPRDTTSTOU',  # Required
    'userId': '1',            # Required
    'languageCode': '1',      # Required (1 for English)
    'Content-Type': 'application/json'
}
"""
            },
            '404': {
                'name': 'Not Found',
                'common_causes': [
                    'Invalid resource ID',
                    'Wrong endpoint path',
                    'Resource has been deleted'
                ],
                'solutions': [
                    'Verify the resource ID exists',
                    'Check endpoint path for typos',
                    'Ensure resource hasn\'t been deleted'
                ],
                'code_snippet': """
# Verify resource exists before operations
def check_resource_exists(resource_id: str) -> bool:
    try:
        response = client.get(f"/resource/{resource_id}")
        return response.status_code == 200
    except:
        return False

if not check_resource_exists(resource_id):
    print(f"Resource {resource_id} not found")
"""
            },
            '429': {
                'name': 'Rate Limit Exceeded',
                'common_causes': [
                    'Too many requests in short time',
                    'Exceeded daily/hourly limits'
                ],
                'solutions': [
                    'Implement exponential backoff',
                    'Add rate limiting to your client',
                    'Cache responses when possible'
                ],
                'code_snippet': """
# Exponential backoff for rate limiting
import time
import random

def retry_with_backoff(func, max_retries=5):
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitException:
            if attempt == max_retries - 1:
                raise
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(wait_time)
"""
            },
            '500': {
                'name': 'Internal Server Error',
                'common_causes': [
                    'Server-side issue',
                    'Temporary service disruption'
                ],
                'solutions': [
                    'Retry with exponential backoff',
                    'Check service status page',
                    'Contact support if persistent'
                ],
                'code_snippet': """
# Handling server errors with retry
async def call_with_retry(func, *args, **kwargs):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except ServerError as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
    return None
"""
            }
        }
        
        error_info = error_solutions.get(error_code, {
            'name': 'Unknown Error',
            'common_causes': ['Error code not recognized'],
            'solutions': ['Check API documentation', 'Contact support'],
            'code_snippet': ''
        })
        
        # Use BERT to analyze the error context
        query_embeddings = self._generate_embeddings(query)
        with torch.no_grad():
            error_pattern = self.error_analyzer(query_embeddings)
        
        return {
            'error_analysis': {
                'error_code': error_code,
                'error_name': error_info['name'],
                'common_causes': error_info['common_causes'],
                'solutions': error_info['solutions']
            },
            'debugging_steps': [
                '1. Check request format and parameters',
                '2. Verify authentication headers',
                '3. Validate data types and formats',
                '4. Check API rate limits',
                '5. Review API documentation for changes'
            ],
            'code_snippet': error_info['code_snippet'],
            'related_documentation': [
                '/docs/error-handling',
                '/docs/authentication',
                '/docs/rate-limiting'
            ]
        }
    
    def _detect_language_preference(self, entities: Dict[str, List[str]]) -> str:
        """Detect programming language preference from entities"""
        if 'PROGRAMMING_LANGUAGE' in entities:
            lang = entities['PROGRAMMING_LANGUAGE'][0].lower()
            if lang in ['python', 'javascript', 'java', 'curl', 'typescript', 'go', 'ruby']:
                return lang
        return self.user_preferences.get('preferred_language', 'python')
    
    def _generate_suggestions(self, intent: QueryIntent, apis: List[APIMatch]) -> List[str]:
        """Generate contextual suggestions for next steps"""
        suggestions = []
        
        if intent.primary_intent == 'get_example':
            suggestions.extend([
                "Try the example with your API credentials",
                "Modify parameters to suit your use case",
                "Add error handling for production use"
            ])
        
        elif intent.primary_intent == 'debug_error':
            suggestions.extend([
                "Check the request/response logs",
                "Verify your authentication setup",
                "Test with minimal parameters first"
            ])
        
        elif intent.primary_intent == 'find_api':
            if apis:
                suggestions.extend([
                    f"Try the {apis[0].endpoint_name} endpoint",
                    "Request code examples for this API",
                    "Check rate limits and quotas"
                ])
        
        return suggestions
    
    def _generate_general_help(self, query: str, relevant_apis: List[APIMatch]) -> Dict[str, Any]:
        """Generate a general help response"""
        message = f"I can help you with your query: '{query}'\n\n"
        
        if relevant_apis:
            message += "Here are some relevant API endpoints:\n"
            for api in relevant_apis[:3]:
                message += f"• {api.endpoint_name}: {api.method} {api.path}\n"
            message += "\nWould you like code examples for any of these endpoints?"
        else:
            message += "Available actions:\n"
            message += "• Get code examples for API endpoints\n"
            message += "• Debug API errors (401, 404, 500, etc.)\n"
            message += "• Find specific APIs for your use case\n"
            message += "• Learn about authentication and best practices\n"
        
        return {
            'message': message,
            'suggestions': [
                "Ask for Python/JavaScript code examples",
                "Describe the error you're facing",
                "Specify which API operation you need"
            ]
        }

    def _describe_apis(self, relevant_apis: List[APIMatch]) -> Dict[str, Any]:
        """Describe the found APIs"""
        if not relevant_apis:
            return {
                'message': "I couldn't find any matching APIs for your query. Try being more specific or check the available endpoints.",
                'suggestions': ["List all endpoints", "Search for account APIs", "Search for customer APIs"]
            }
        
        message = f"I found {len(relevant_apis)} relevant API endpoint(s):\n\n"
        
        for i, api in enumerate(relevant_apis[:5], 1):
            message += f"{i}. **{api.endpoint_name}**\n"
            message += f"   Method: {api.method}\n"
            message += f"   Path: {api.path}\n"
            message += f"   Confidence: {api.confidence:.2%}\n"
            
            if api.parameters:
                message += f"   Parameters: {len(api.parameters)}\n"
            message += "\n"
        
        return {
            'message': message,
            'related_apis': [
                {
                    'name': api.endpoint_name,
                    'method': api.method,
                    'path': api.path,
                    'confidence': api.confidence
                }
                for api in relevant_apis[:5]
            ],
            'suggestions': [
                f"Get code example for {relevant_apis[0].endpoint_name}" if relevant_apis else None,
                "View parameter details",
                "See authentication requirements"
            ]
        }

    def _explain_parameters(self, relevant_apis: List[APIMatch]) -> Dict[str, Any]:
        """Explain parameters for the matched APIs"""
        if not relevant_apis:
            return {
                'message': "No APIs found to explain parameters for.",
                'suggestions': ["Search for specific API", "List all endpoints"]
            }
        
        api = relevant_apis[0]  # Focus on the best match
        message = f"**Parameters for {api.endpoint_name}**\n\n"
        message += f"Endpoint: {api.method} {api.path}\n\n"
        
        if not api.parameters:
            message += "This endpoint doesn't require any parameters.\n"
        else:
            message += "Required Parameters:\n"
            required_params = [p for p in api.parameters if p.get('required', False)]
            optional_params = [p for p in api.parameters if not p.get('required', False)]
            
            if required_params:
                for param in required_params:
                    message += f"• **{param.get('name', 'unknown')}** ({param.get('type', 'string')})\n"
                    if param.get('description'):
                        message += f"  {param['description']}\n"
            else:
                message += "  None\n"
            
            if optional_params:
                message += "\nOptional Parameters:\n"
                for param in optional_params:
                    message += f"• {param.get('name', 'unknown')} ({param.get('type', 'string')})\n"
                    if param.get('description'):
                        message += f"  {param['description']}\n"
        
        return {
            'message': message,
            'parameters': self._format_parameters(api.parameters),
            'suggestions': [
                f"Get code example for {api.endpoint_name}",
                "View response format",
                "See error codes"
            ]
        }

    def _provide_best_practices(self, relevant_apis: List[APIMatch]) -> Dict[str, Any]:
        """Provide best practices for API usage"""
        message = "**Best Practices for BAnCS API Usage**\n\n"
        
        message += "**1. Authentication**\n"
        message += "Always include these headers in every request:\n"
        message += "```\n"
        message += "entity: GPRDTTSTOU\n"
        message += "userId: 1\n"
        message += "languageCode: 1\n"
        message += "```\n\n"
        
        message += "**2. Error Handling**\n"
        message += "• Implement retry logic with exponential backoff\n"
        message += "• Handle specific error codes (400, 401, 404, 429, 500)\n"
        message += "• Log all requests and responses for debugging\n\n"
        
        message += "**3. Rate Limiting**\n"
        message += "• Respect rate limits (typically 60 requests/minute)\n"
        message += "• Implement request queuing for bulk operations\n"
        message += "• Cache responses when appropriate\n\n"
        
        message += "**4. Data Validation**\n"
        message += "• Validate input data before sending requests\n"
        message += "• Use proper date format (YYYYMMDD for BAnCS)\n"
        message += "• Check account/customer IDs exist before operations\n\n"
        
        message += "**5. Security**\n"
        message += "• Never hardcode credentials\n"
        message += "• Use HTTPS for all requests\n"
        message += "• Implement proper session management\n"
        
        if relevant_apis:
            message += f"\n\nFor your specific use case ({relevant_apis[0].endpoint_name}), "
            message += "remember to validate all required parameters before making the request."
        
        return {
            'message': message,
            'suggestions': [
                "Get code example with error handling",
                "View rate limit details",
                "See authentication setup"
            ]
        }

    def _compare_apis(self, relevant_apis: List[APIMatch]) -> Dict[str, Any]:
        """Compare multiple APIs"""
        if len(relevant_apis) < 2:
            return {
                'message': "I need at least 2 APIs to compare. Please be more specific about which APIs you want to compare.",
                'suggestions': ["List all account APIs", "List all customer APIs", "Show all endpoints"]
            }
        
        message = "**API Comparison**\n\n"
        
        # Compare first two APIs
        for i, api in enumerate(relevant_apis[:2], 1):
            message += f"**{i}. {api.endpoint_name}**\n"
            message += f"• Method: {api.method}\n"
            message += f"• Path: {api.path}\n"
            message += f"• Parameters: {len(api.parameters)}\n"
            message += f"• Match confidence: {api.confidence:.2%}\n\n"
        
        # Highlight differences
        api1, api2 = relevant_apis[0], relevant_apis[1]
        message += "**Key Differences:**\n"
        
        if api1.method != api2.method:
            message += f"• Different HTTP methods: {api1.method} vs {api2.method}\n"
        
        if len(api1.parameters) != len(api2.parameters):
            message += f"• Different parameter counts: {len(api1.parameters)} vs {len(api2.parameters)}\n"
        
        # Common use cases
        message += "\n**When to use each:**\n"
        if 'GET' in api1.method:
            message += f"• Use {api1.endpoint_name} for retrieving data\n"
        if 'POST' in api1.method:
            message += f"• Use {api1.endpoint_name} for creating new resources\n"
        if 'GET' in api2.method:
            message += f"• Use {api2.endpoint_name} for retrieving data\n"
        if 'POST' in api2.method:
            message += f"• Use {api2.endpoint_name} for creating new resources\n"
        
        return {
            'message': message,
            'comparison': {
                'api1': {'name': api1.endpoint_name, 'method': api1.method, 'path': api1.path},
                'api2': {'name': api2.endpoint_name, 'method': api2.method, 'path': api2.path}
            },
            'suggestions': [
                f"Get code example for {api1.endpoint_name}",
                f"Get code example for {api2.endpoint_name}",
                "View detailed parameter comparison"
            ]
        }

    def _format_parameters(self, parameters: List[Dict]) -> List[Dict[str, Any]]:
        """Format parameters for response"""
        if not parameters:
            return []
        
        formatted = []
        for param in parameters[:20]:  # Limit to 20 parameters
            formatted_param = {
                'name': param.get('name', 'unknown'),
                'type': param.get('type', 'string'),
                'required': param.get('required', False),
                'description': param.get('description', ''),
                'in': param.get('in', 'query')  # where the parameter goes (path, query, header, body)
            }
            
            # Add additional properties if they exist
            if 'default' in param:
                formatted_param['default'] = param['default']
            if 'enum' in param:
                formatted_param['enum'] = param['enum']
            if 'example' in param:
                formatted_param['example'] = param['example']
            
            formatted.append(formatted_param)
        
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
            'description': 'BAnCS requires custom headers for authentication. Include these in every request.',
            'example': {
                'python': "headers = {'entity': 'GPRDTTSTOU', 'userId': '1', 'languageCode': '1'}",
                'javascript': "const headers = {entity: 'GPRDTTSTOU', userId: '1', languageCode: '1'}",
                'curl': "-H 'entity: GPRDTTSTOU' -H 'userId: 1' -H 'languageCode: 1'"
            }
        }

    def _get_rate_limit_info(self, api: APIMatch) -> Dict[str, Any]:
        """Get rate limit information for an API"""
        # Default rate limits for BAnCS APIs
        return {
            'requests_per_minute': 60,
            'requests_per_hour': 1000,
            'requests_per_day': 10000,
            'retry_after_header': True,
            'burst_limit': 10,
            'recommendation': 'Implement exponential backoff with jitter for optimal performance',
            'example_retry': """
import time
import random

def retry_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(wait_time)
"""
        }

    # Utility methods for code generation
    def _to_camel_case(self, text: str) -> str:
        """Convert to camelCase"""
        words = text.replace(' ', '_').split('_')
        return words[0].lower() + ''.join(word.capitalize() for word in words[1:])
    
    def _to_pascal_case(self, text: str) -> str:
        """Convert to PascalCase"""
        words = text.replace(' ', '_').split('_')
        return ''.join(word.capitalize() for word in words)
    
    def _get_python_type(self, param: Dict) -> str:
        """Get Python type from OpenAPI type"""
        type_map = {
            'string': 'str',
            'integer': 'int',
            'number': 'float',
            'boolean': 'bool',
            'array': 'List[Any]',
            'object': 'Dict[str, Any]'
        }
        return type_map.get(param.get('type', 'string'), 'Any')
    
    def _generate_function_params(self, parameters: List[Dict]) -> str:
        """Generate function parameters for Python"""
        if not parameters:
            return ""
        
        params = []
        for param in parameters:
            param_name = param.get('name', 'param')
            param_type = self._get_python_type(param)
            
            if param.get('required', False):
                params.append(f"{param_name}: {param_type}")
            else:
                default_value = param.get('default', 'None')
                if default_value != 'None' and param_type == 'str':
                    default_value = f'"{default_value}"'
                params.append(f"{param_name}: Optional[{param_type}] = {default_value}")
        
        return ',\n    '.join(params)

    def _generate_js_params(self, parameters: List[Dict]) -> str:
        """Generate function parameters for JavaScript"""
        if not parameters:
            return ""
        
        # For JavaScript, we'll use destructuring with defaults
        param_names = []
        for param in parameters:
            param_name = param.get('name', 'param')
            if param.get('required', False):
                param_names.append(param_name)
            else:
                default_val = param.get('default', 'null')
                param_names.append(f"{param_name} = {default_val}")
        
        return '{' + ', '.join(param_names) + '}'

    def _generate_request_body(self, api: APIMatch, language: str) -> str:
        """Generate request body code for different languages"""
        body_params = [p for p in api.parameters if p.get('in', 'query') == 'body']
        
        if not body_params:
            return ""
        
        if language == 'python':
            return f"""
    request_body = {{
        {', '.join([f'"{p["name"]}": {p["name"]}' for p in body_params])}
    }}"""
        elif language == 'javascript':
            return f"""
    const requestBody = {{
        {', '.join([f'{p["name"]}' for p in body_params])}
    }};"""
        elif language == 'java':
            return f"""
        Map<String, Object> requestBody = new HashMap<>();
        {' '.join([f'requestBody.put("{p["name"]}", {p["name"]});' for p in body_params])}"""
        
        return ""

    def _generate_request_params(self, api: APIMatch, language: str) -> str:
        """Generate request parameters code"""
        if language == 'python':
            return "json=request_body" if any(p.get('in') == 'body' for p in api.parameters) else "params=params"
        elif language == 'javascript':
            return "data: requestBody" if any(p.get('in') == 'body' for p in api.parameters) else "params: params"
        return ""

    def _generate_example_params(self, parameters: List[Dict], language: str = 'python') -> str:
        """Generate example parameter values"""
        if not parameters:
            return ""
        
        examples = []
        for param in parameters:
            if param.get('required', False):
                param_name = param.get('name', 'param')
                if param.get('example'):
                    value = param['example']
                elif param.get('type') == 'integer':
                    value = 1
                elif param.get('type') == 'boolean':
                    value = 'true' if language == 'javascript' else 'True'
                else:
                    value = f'"example_{param_name}"'
                
                if language == 'python':
                    examples.append(f"{param_name}={value}")
                elif language == 'javascript':
                    examples.append(f"{param_name}: {value}")
        
        return ', '.join(examples)

    def _generate_curl_data(self, api: APIMatch) -> str:
        """Generate cURL data parameter"""
        body_params = [p for p in api.parameters if p.get('in', 'query') == 'body']
        
        if not body_params:
            return ""
        
        sample_data = {}
        for param in body_params:
            if param.get('example'):
                sample_data[param['name']] = param['example']
            elif param.get('type') == 'integer':
                sample_data[param['name']] = 1
            elif param.get('type') == 'boolean':
                sample_data[param['name']] = True
            else:
                sample_data[param['name']] = f"example_{param['name']}"
        
        return f"-d '{json.dumps(sample_data, indent=2)}'"

    def _generate_java_params(self, parameters: List[Dict]) -> str:
        """Generate Java method parameters"""
        if not parameters:
            return ""
        
        params = []
        for param in parameters:
            if param.get('required', False):
                java_type = self._get_java_type(param)
                params.append(f"{java_type} {param['name']}")
        
        return ', '.join(params)

    def _generate_java_request_builder(self, api: APIMatch) -> str:
        """Generate Java request builder code"""
        if any(p.get('in') == 'body' for p in api.parameters):
            return ".post(RequestBody.create(MediaType.parse(\"application/json\"), mapper.writeValueAsString(requestBody)))"
        else:
            return f".{api.method.lower()}()"

    def _get_java_type(self, param: Dict) -> str:
        """Get Java type from parameter definition"""
        type_map = {
            'string': 'String',
            'integer': 'Integer',
            'number': 'Double',
            'boolean': 'Boolean',
            'array': 'List<Object>',
            'object': 'Map<String, Object>'
        }
        return type_map.get(param.get('type', 'string'), 'Object')
    
    async def _generate_response(
        self,
        query: str,
        intent: QueryIntent,
        entities: Dict[str, List[str]],
        relevant_apis: List[APIMatch],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comprehensive response based on analysis"""
        
        response = {
            'status': 'success',
            'intent': intent.primary_intent,
            'confidence': intent.confidence,
            'response_type': 'comprehensive'
        }
        
        # Handle different intents
        if intent.primary_intent == 'get_example':
            response.update(self._generate_code_example(relevant_apis, entities))
        
        elif intent.primary_intent == 'debug_error':
            response.update(await self._analyze_error(query, entities, context))
        
        elif intent.primary_intent == 'find_api':
            response.update(self._describe_apis(relevant_apis))
        
        elif intent.primary_intent == 'understand_params':
            response.update(self._explain_parameters(relevant_apis))
        
        elif intent.primary_intent == 'best_practices':
            response.update(self._provide_best_practices(relevant_apis))
        
        elif intent.primary_intent == 'compare_apis':
            response.update(self._compare_apis(relevant_apis))
        
        else:
            # General response
            response.update(self._generate_general_help(query, relevant_apis))
        
        # Add contextual suggestions
        response['suggestions'] = self._generate_suggestions(intent, relevant_apis)
        
        # Add related APIs
        response['related_apis'] = [
            {
                'name': api.endpoint_name,
                'confidence': api.confidence,
                'method': api.method,
                'path': api.path
            }
            for api in relevant_apis[:3]
        ]
        
        return response

    # Public wrapper methods for compatibility
    def generate_code_example(self, endpoint_name: str, endpoint_metadata: Dict) -> List[Dict]:
        """Public wrapper for code generation"""
        # Create a dummy APIMatch object for the private method
        api_match = APIMatch(
            endpoint_name=endpoint_name,
            method=endpoint_metadata.get('method', 'GET'),
            path=endpoint_metadata.get('path', '/'),
            confidence=1.0,
            parameters=endpoint_metadata.get('parameters', [])
        )
        result = self._generate_code_example([api_match], {})
        return result.get('code_examples', [])
    
    def analyze_error(self, query: str, error_info: str) -> Dict:
        """Public wrapper for error analysis"""
        entities = {'ERROR_CODE': [error_info]}
        # Run the async method synchronously
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self._analyze_error(query, entities, None))
            return result.get('error_analysis', {})
        finally:
            loop.close()