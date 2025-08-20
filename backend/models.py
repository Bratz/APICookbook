# backend/models.py - Minimal version
from pydantic import BaseModel,Field, validator
from typing import Dict, Any, Optional, List, Union

class ChatMessage(BaseModel):
    message: str
    context: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = Field(default_factory=list)
    
    @validator('context', pre=True)
    def validate_context(cls, v):
        """Ensure context is in the right format"""
        if v is None:
            return []
        if isinstance(v, dict):
            return [v]  # Convert single dict to list
        if isinstance(v, list):
            return v
        return []
    
    def get_context_list(self) -> List[Dict[str, Any]]:
        """Get context as a list"""
        if isinstance(self.context, dict):
            return [self.context]
        elif isinstance(self.context, list):
            return self.context
        return []

class ChatResponse(BaseModel):
    response: str
    confidence: float = 0.8
    mcp_available: bool = False
    api_references: List[Dict[str, Any]] = []

class CookbookRecipe(BaseModel):
    id: str
    title: str
    description: str
    category: str
    method: str
    url: str