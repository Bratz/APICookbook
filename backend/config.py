# backend/config.py
"""
Configuration management for TCS BAnCS API Cookbook
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
        default="TCS BAnCS API MCP Server",
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
        description="BAnCS API base URL"
    )
    api_timeout: int = Field(
        default=30,
        description="API request timeout in seconds"
    )
    api_retry_attempts: int = Field(
        default=3,
        description="Number of retry attempts for API calls"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key if required"
    )
    
    # ===================== API Discovery Settings =====================
    api_discovery_enabled: bool = Field(
        default=False,
        description="Enable automatic API discovery from live endpoints"
    )
    api_spec_auto_reload: bool = Field(
        default=True,
        description="Automatically reload API specs when changed"
    )
    api_spec_watch_interval: int = Field(
        default=60,
        description="Interval in seconds to check for spec changes"
    )
    
    # ===================== Server Settings =====================
    host: str = Field(
        default="0.0.0.0",
        description="Server host"
    )
    api_port: int = Field(
        default=7600,
        description="API server port"
    )
    mcp_port: int = Field(
        default=7650,
        description="MCP server port"
    )
    
    # ===================== BAnCS Specific Settings =====================
    bancs_entity: str = Field(
        default="GPRDTTSTOU",
        description="BAnCS entity code"
    )
    bancs_user_id: str = Field(
        default="1",
        description="BAnCS user ID"
    )
    bancs_language_code: str = Field(
        default="1",
        description="BAnCS language code (1 for English)"
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
    max_search_results: int = Field(
        default=5,
        description="Maximum number of search results"
    )
    
    # ===================== MCP Settings =====================
    mcp_enabled: bool = Field(
        default=True,
        description="Enable MCP server and tools"
    )
    mcp_server_name: str = Field(
        default="tcs-bancs-mcp",
        description="MCP server name"
    )
    mcp_max_concurrent_tools: int = Field(
        default=10,
        description="Maximum concurrent MCP tool executions"
    )
    
    # ===================== Cache Settings =====================
    cache_enabled: bool = Field(
        default=True,
        description="Enable response caching"
    )
    cache_ttl: int = Field(
        default=300,
        description="Cache TTL in seconds"
    )
    cache_max_size: int = Field(
        default=1000,
        description="Maximum cache size (number of entries)"
    )
    
    # ===================== Rate Limiting =====================
    rate_limit_enabled: bool = Field(
        default=False,
        description="Enable rate limiting"
    )
    rate_limit_requests: int = Field(
        default=100,
        description="Maximum requests per window"
    )
    rate_limit_window: int = Field(
        default=60,
        description="Rate limit window in seconds"
    )
    
    # ===================== Database Settings =====================
    database_url: Optional[str] = Field(
        default=None,
        description="Database connection URL"
    )
    redis_url: Optional[str] = Field(
        default=None,
        description="Redis connection URL for caching"
    )
    redis_enabled: bool = Field(
        default=False,
        description="Enable Redis caching"
    )
    
    # ===================== Security Settings =====================
    jwt_secret_key: Optional[str] = Field(
        default="your-secret-key-change-in-production",
        description="JWT secret key for token generation"
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT algorithm"
    )
    jwt_expiration_minutes: int = Field(
        default=30,
        description="JWT token expiration in minutes"
    )
    api_key_enabled: bool = Field(
        default=False,
        description="Enable API key authentication"
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
    log_file_enabled: bool = Field(
        default=False,
        description="Enable file logging"
    )
    log_file_path: str = Field(
        default="logs/app.log",
        description="Log file path"
    )
    log_file_rotation: str = Field(
        default="midnight",
        description="Log file rotation schedule"
    )
    log_file_retention: int = Field(
        default=7,
        description="Log file retention in days"
    )
    
    # ===================== Feature Flags =====================
    feature_chat_enabled: bool = Field(
        default=True,
        description="Enable chat functionality"
    )
    feature_cookbook_enabled: bool = Field(
        default=True,
        description="Enable cookbook interface"
    )
    feature_playground_enabled: bool = Field(
        default=True,
        description="Enable API playground"
    )
    feature_websocket_enabled: bool = Field(
        default=True,
        description="Enable WebSocket support"
    )
    
    # ===================== LLM Settings =====================
    llm_enabled: bool = Field(
        default=False,
        description="Enable LLM features"
    )
    llm_provider: str = Field(
        default="local",
        description="LLM provider (local, ollama, openai)"
    )
    llm_model: str = Field(
        default="microsoft/codebert-base",
        description="Model to use for chatbot"
    )
    llm_max_tokens: int = Field(
        default=2048,
        description="Maximum tokens for response"
    )
    llm_temperature: float = Field(
        default=0.7,
        description="Temperature for generation"
    )
    llm_device: str = Field(
        default="cpu",
        description="Device to run model on (cpu, cuda, mps)"
    )
    llm_context_window: int = Field(
        default=8192,
        description="Context window size"
    )
    llm_use_flash_attention: bool = Field(
        default=False,
        description="Use Flash Attention for efficiency"
    )
    llm_use_tool_calling: bool = Field(
        default=True,
        description="Enable function/tool calling"
    )
    llm_quantization: str = Field(
        default="Q4_K_M",
        description="Quantization level"
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
    rag_chunk_size: int = Field(
        default=512,
        description="Chunk size for document splitting"
    )
    rag_top_k: int = Field(
        default=5,
        description="Number of relevant chunks to retrieve"
    )
    
    # ===================== Monitoring =====================
    metrics_enabled: bool = Field(
        default=False,
        description="Enable metrics collection"
    )
    metrics_port: int = Field(
        default=9090,
        description="Metrics server port"
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
    
    # ===================== Computed Properties =====================
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == "development"
    
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

# Test the configuration if running directly
if __name__ == "__main__":
    print("Settings loaded successfully!")
    print(f"App Name: {settings.app_name}")
    print(f"API Port: {settings.api_port}")
    print(f"API Discovery: {settings.api_discovery_enabled}")
    print(f"Cache TTL: {settings.cache_ttl}")
    print(f"MCP Enabled: {settings.mcp_enabled}")
    print("\nAll attributes:")
    for key, value in settings.model_dump().items():
        print(f"  {key}: {value}")