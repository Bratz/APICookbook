# backend/utils.py
"""
Utility functions for TCS BAnCS API Cookbook
"""

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import jwt
from functools import wraps
import asyncio
import time

from config import settings

# ===================== String Utilities =====================

def to_camel_case(snake_str: str) -> str:
    """Convert snake_case to camelCase"""
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

def to_snake_case(camel_str: str) -> str:
    """Convert camelCase or PascalCase to snake_case"""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_str)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def to_pascal_case(snake_str: str) -> str:
    """Convert snake_case to PascalCase"""
    return ''.join(word.title() for word in snake_str.split('_'))

def sanitize_string(text: str, max_length: int = 200) -> str:
    """Sanitize and truncate string"""
    # Remove special characters
    sanitized = re.sub(r'[^\w\s-]', '', text)
    # Truncate if needed
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length-3] + "..."
    return sanitized

# ===================== Date/Time Utilities =====================

def format_bancs_date(date: datetime) -> str:
    """Format date for BAnCS API (YYYYMMDD)"""
    return date.strftime("%Y%m%d")

def parse_bancs_date(date_str: str) -> datetime:
    """Parse BAnCS date format (YYYYMMDD)"""
    return datetime.strptime(date_str, "%Y%m%d")

def get_current_bancs_date() -> str:
    """Get current date in BAnCS format"""
    return format_bancs_date(datetime.now())

def add_business_days(start_date: datetime, days: int) -> datetime:
    """Add business days to a date (excluding weekends)"""
    current = start_date
    while days > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Monday = 0, Friday = 4
            days -= 1
    return current

# ===================== Validation Utilities =====================

def validate_account_reference(account_ref: str) -> bool:
    """Validate account reference format (15 digits)"""
    return bool(re.match(r'^\d{15}$', account_ref))

def validate_currency_code(currency: str) -> bool:
    """Validate currency code format (3 uppercase letters)"""
    return bool(re.match(r'^[A-Z]{3}$', currency))

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

def validate_phone(phone: str) -> bool:
    """Validate phone number (10-15 digits)"""
    cleaned = re.sub(r'[^\d]', '', phone)
    return 10 <= len(cleaned) <= 15

def validate_amount(amount: Union[float, str]) -> bool:
    """Validate monetary amount"""
    try:
        value = float(amount)
        return value > 0 and value < 1000000000  # Max 1 billion
    except (ValueError, TypeError):
        return False

# ===================== Security Utilities =====================

def generate_api_key() -> str:
    """Generate a secure API key"""
    import secrets
    return secrets.token_urlsafe(32)

def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    import bcrypt
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against a hash"""
    import bcrypt
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def generate_jwt_token(payload: Dict[str, Any]) -> str:
    """Generate a JWT token"""
    payload['exp'] = datetime.utcnow() + timedelta(minutes=settings.jwt_expiration_minutes)
    payload['iat'] = datetime.utcnow()
    
    return jwt.encode(
        payload,
        settings.jwt_secret_key or "default-secret-key",
        algorithm=settings.jwt_algorithm
    )

def decode_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and verify a JWT token"""
    try:
        return jwt.decode(
            token,
            settings.jwt_secret_key or "default-secret-key",
            algorithms=[settings.jwt_algorithm]
        )
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

# ===================== Caching Utilities =====================

class SimpleCache:
    """Simple in-memory cache with TTL"""
    
    def __init__(self, ttl: int = 300, max_size: int = 1000):
        self.cache = {}
        self.ttl = ttl
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if (datetime.now() - timestamp).seconds < self.ttl:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        # Enforce max size
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (value, datetime.now())
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "entries": len(self.cache),
            "max_size": self.max_size,
            "ttl": self.ttl
        }

# ===================== Rate Limiting =====================

class RateLimiter:
    """Simple rate limiter"""
    
    def __init__(self, requests: int = 100, window: int = 60):
        self.requests = requests
        self.window = window
        self.clients = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        now = time.time()
        
        if client_id not in self.clients:
            self.clients[client_id] = []
        
        # Remove old requests
        self.clients[client_id] = [
            timestamp for timestamp in self.clients[client_id]
            if now - timestamp < self.window
        ]
        
        # Check limit
        if len(self.clients[client_id]) < self.requests:
            self.clients[client_id].append(now)
            return True
        
        return False
    
    def reset(self, client_id: str):
        """Reset rate limit for a client"""
        if client_id in self.clients:
            del self.clients[client_id]

# ===================== Decorators =====================

def async_retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator for async function retry"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay * (2 ** attempt))
            raise last_exception
        return wrapper
    return decorator

def measure_time(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.2f} seconds")
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.2f} seconds")
        return result
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

# ===================== Data Transformation =====================

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """Unflatten dictionary"""
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result

def mask_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Mask sensitive information in dictionary"""
    sensitive_fields = [
        'password', 'token', 'api_key', 'secret', 
        'account_number', 'ssn', 'credit_card'
    ]
    
    masked = data.copy()
    for key, value in masked.items():
        if any(field in key.lower() for field in sensitive_fields):
            if isinstance(value, str) and len(value) > 4:
                masked[key] = value[:2] + '*' * (len(value) - 4) + value[-2:]
            else:
                masked[key] = '***'
        elif isinstance(value, dict):
            masked[key] = mask_sensitive_data(value)
    
    return masked

# ===================== File Utilities =====================

def generate_file_hash(content: bytes) -> str:
    """Generate SHA256 hash of file content"""
    return hashlib.sha256(content).hexdigest()

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

# ===================== API Response Formatting =====================

def format_api_response(
    data: Any = None,
    message: str = "Success",
    status: str = "success",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Format standardized API response"""
    response = {
        "status": status,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    if data is not None:
        response["data"] = data
    
    if metadata:
        response["metadata"] = metadata
    
    return response

def format_error_response(
    error: str,
    code: Optional[str] = None,
    details: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Format standardized error response"""
    response = {
        "status": "error",
        "error": error,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    if code:
        response["code"] = code
    
    if details:
        response["details"] = details
    
    return response

# ===================== Pagination Utilities =====================

def paginate_results(
    items: List[Any],
    page: int = 1,
    page_size: int = 20
) -> Dict[str, Any]:
    """Paginate a list of items"""
    total = len(items)
    total_pages = (total + page_size - 1) // page_size
    
    start = (page - 1) * page_size
    end = start + page_size
    
    return {
        "items": items[start:end],
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
    }

# ===================== Export All Utilities =====================

__all__ = [
    # String utilities
    "to_camel_case",
    "to_snake_case",
    "to_pascal_case",
    "sanitize_string",
    
    # Date/Time utilities
    "format_bancs_date",
    "parse_bancs_date",
    "get_current_bancs_date",
    "add_business_days",
    
    # Validation utilities
    "validate_account_reference",
    "validate_currency_code",
    "validate_email",
    "validate_phone",
    "validate_amount",
    
    # Security utilities
    "generate_api_key",
    "hash_password",
    "verify_password",
    "generate_jwt_token",
    "decode_jwt_token",
    
    # Caching
    "SimpleCache",
    
    # Rate limiting
    "RateLimiter",
    
    # Decorators
    "async_retry",
    "measure_time",
    
    # Data transformation
    "flatten_dict",
    "unflatten_dict",
    "mask_sensitive_data",
    
    # File utilities
    "generate_file_hash",
    "format_file_size",
    
    # API response formatting
    "format_api_response",
    "format_error_response",
    
    # Pagination
    "paginate_results"
]