"""
Main entry point for the LLM inference engine.
"""
import uvicorn
import logging
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info(f"Starting LLM Inference Engine on {settings.api_host}:{settings.api_port}")
    
    uvicorn.run(
        "api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=1,  # Single worker for shared model instance
        log_level=settings.log_level.lower()
    )
