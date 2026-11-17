# backend/config.py
"""
Configuration management for TCS BaNCS API Cookbook
Complete version with ALL required attributes
"""

from typing import List, Optional, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
import os

class Settings(BaseSettings):
    """Application settings with all required attributes"""
    
    # ===================== Application Settings =====================
    app_name: str = Field(
        default="TCS BaNCS API MCP Server",
        description="Application name"
    )
    app_version: str = Field(
        default="1.0.0",
        description="Application version"
    )
    debug: bool = Field(
        default=False,
        description="Debug mode"
    )
    environment: str = Field(
        default="development",
        description="Environment (development, staging, production)"
    )
    
    # ===================== API Settings =====================
    api_base_url: str = Field(
        default="https://demoapps.tcsbancs.com",
        description="BaNCS API base URL"
    )
    api_timeout: int = Field(
        default=30,
        description="API request timeout in seconds"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key if required"
    )
    
    # ===================== Server Settings =====================
    host: str = Field(
        default="127.0.0.1",  # no auth on any route - set HOST=0.0.0.0 only behind a trusted network/proxy
        description="Server host"
    )
    api_port: int = Field(
        default=7600,
        description="API server port"
    )
    
    # ===================== BaNCS Specific Settings =====================
    bancs_entity: str = Field(
        default="GPRDTTSTOU",
        description="BaNCS entity code"
    )
    bancs_user_id: str = Field(
        default="1",
        description="BaNCS user ID"
    )
    bancs_language_code: str = Field(
        default="1",
        description="BaNCS language code (1 for English)"
    )
    bancs_token_required: bool = Field(
        default=False,
        description="Calls that reach the bank need the visitor's BaNCS access token (Accesstoken header)"
    )
    
    # ===================== CORS Settings =====================
    cors_origins: List[str] = Field(
        default=[
            "http://localhost:7500",
            "http://localhost:7600",
            "http://127.0.0.1:7500",
            "http://127.0.0.1:7600"
        ],
        description="Allowed CORS origins"
    )
    cors_allow_credentials: bool = Field(
        default=True,
        description="Allow credentials in CORS"
    )
    cors_allow_methods: List[str] = Field(
        default=["*"],
        description="Allowed HTTP methods"
    )
    cors_allow_headers: List[str] = Field(
        default=["*"],
        description="Allowed headers"
    )
    
    # ===================== Search Settings =====================
    search_threshold: int = Field(
        default=60,
        description="Minimum score threshold for search results"
    )
    
    # ===================== MCP Settings =====================
    mcp_enabled: bool = Field(
        default=True,
        description="Enable MCP server and tools"
    )
    
    hoppscotch_enabled: bool = Field(
        default=True,  # read by /health; was only ever set by .env, so it crashed without one (Docker)
        description="Enable Hoppscotch export"
    )

    # ===================== Cache Settings =====================
    cache_ttl: int = Field(
        default=300,
        description="Cache TTL in seconds"
    )
    
    # ===================== Logging Settings =====================
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    log_file: Optional[str] = Field(
        default=None,
        description="Log file path"
    )
    
    # ===================== RAG Settings =====================
    rag_enabled: bool = Field(
        default=False,
        description="Enable RAG for accurate responses"
    )
    rag_embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Embedding model for RAG"
    )
    
    # ===================== Field Validators for Pydantic v2 =====================
    
    @field_validator('cors_origins', mode='before')
    @classmethod
    def parse_cors_origins(cls, v: Any) -> List[str]:
        """Parse CORS origins from string or list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',') if origin.strip()]
        elif isinstance(v, list):
            return v
        else:
            return ["http://localhost:7600"]
    
    @field_validator('cors_allow_methods', mode='before')
    @classmethod
    def parse_cors_methods(cls, v: Any) -> List[str]:
        """Parse CORS methods from string or list"""
        if isinstance(v, str):
            if v == "*":
                return ["*"]
            return [method.strip() for method in v.split(',') if method.strip()]
        elif isinstance(v, list):
            return v
        else:
            return ["*"]
    
    @field_validator('cors_allow_headers', mode='before')
    @classmethod
    def parse_cors_headers(cls, v: Any) -> List[str]:
        """Parse CORS headers from string or list"""
        if isinstance(v, str):
            if v == "*":
                return ["*"]
            return [header.strip() for header in v.split(',') if header.strip()]
        elif isinstance(v, list):
            return v
        else:
            return ["*"]
    
    # ===================== Model Configuration for Pydantic v2 =====================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
        env_nested_delimiter="__"
    )

# Create singleton instance
try:
    settings = Settings()
except Exception as e:
    print(f"Error loading settings: {e}")
    print("Using default settings...")
    settings = Settings(_env_file=None)
