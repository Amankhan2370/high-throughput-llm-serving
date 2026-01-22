"""
Configuration settings loaded from environment variables.
All sensitive values must be provided via environment variables.
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""
    
    # Model Configuration
    model_name: str = "ADD_YOUR_OWN_MODEL_NAME"
    model_path: Optional[str] = None
    device: str = "cuda"
    max_sequence_length: int = 2048
    batch_size: int = 8
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    workers: int = 4
    max_concurrent_requests: int = 100
    request_timeout: int = 30
    
    # Batching Configuration
    max_batch_size: int = 32
    batch_timeout_ms: int = 50
    dynamic_batching: bool = True
    
    # Caching Configuration
    cache_enabled: bool = True
    cache_type: str = "memory"  # memory or redis
    redis_url: Optional[str] = None
    cache_ttl: int = 3600
    max_cache_size: int = 10000
    
    # Queue Configuration
    max_queue_size: int = 1000
    queue_timeout: int = 60
    
    # Metrics Configuration
    metrics_enabled: bool = True
    metrics_port: int = 9090
    prometheus_enabled: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Performance
    torch_compile: bool = False
    use_fp16: bool = False
    use_bf16: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
