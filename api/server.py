"""
FastAPI server with async request handling, timeouts, and backpressure.
"""
import asyncio
import time
import uuid
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging

from inference.engine import InferenceEngine
from profiling.metrics import MetricsCollector
from config.settings import settings

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM Inference Engine",
    description="Production-ready LLM inference and serving system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized in startup)
inference_engine: Optional[InferenceEngine] = None
metrics_collector: Optional[MetricsCollector] = None
request_semaphore: Optional[asyncio.Semaphore] = None


# Request/Response Models
class InferenceRequest(BaseModel):
    """Inference request model."""
    prompt: str = Field(..., description="Input prompt", min_length=1)
    max_tokens: int = Field(100, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    use_cache: bool = Field(True, description="Use cache if available")
    request_id: Optional[str] = Field(None, description="Optional request ID")


class InferenceResponse(BaseModel):
    """Inference response model."""
    text: str
    request_id: str
    latency_ms: float
    tokens_generated: Optional[int] = None
    cached: bool = False


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    queue_size: int
    metrics: Dict[str, Any]


class MetricsResponse(BaseModel):
    """Metrics response."""
    metrics: Dict[str, Any]


@app.on_event("startup")
async def startup():
    """Initialize services on startup."""
    global inference_engine, metrics_collector, request_semaphore
    
    try:
        from inference.runtime import InferenceRuntime
        from inference.batching import DynamicBatcher
        from inference.cache import Cache
        
        # Initialize runtime
        runtime = InferenceRuntime(
            model_name=settings.model_name,
            model_path=settings.model_path,
            device=settings.device,
            max_sequence_length=settings.max_sequence_length,
            use_fp16=settings.use_fp16,
            use_bf16=settings.use_bf16,
            torch_compile=settings.torch_compile
        )
        
        # Initialize batcher
        batcher = DynamicBatcher(
            max_batch_size=settings.max_batch_size,
            batch_timeout_ms=settings.batch_timeout_ms,
            max_sequence_length=settings.max_sequence_length
        )
        
        # Initialize cache
        cache = None
        if settings.cache_enabled:
            cache = Cache(
                cache_type=settings.cache_type,
                redis_url=settings.redis_url,
                ttl=settings.cache_ttl,
                max_size=settings.max_cache_size
            )
        
        # Initialize engine
        inference_engine = InferenceEngine(
            runtime=runtime,
            batcher=batcher,
            cache=cache,
            cache_enabled=settings.cache_enabled
        )
        
        await inference_engine.start()
        
        # Initialize metrics
        metrics_collector = MetricsCollector()
        if settings.prometheus_enabled:
            metrics_collector.start_prometheus_server(settings.metrics_port)
        
        # Initialize semaphore for concurrency control
        request_semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
        
        logger.info("Server started successfully")
    
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    global inference_engine
    if inference_engine:
        await inference_engine.stop()
    logger.info("Server shutdown complete")


@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    """Timeout middleware."""
    try:
        response = await asyncio.wait_for(
            call_next(request),
            timeout=settings.request_timeout
        )
        return response
    except asyncio.TimeoutError:
        logger.warning(f"Request timeout: {request.url}")
        return JSONResponse(
            status_code=504,
            content={"error": "Request timeout"}
        )


@app.post("/api/v1/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest, background_tasks: BackgroundTasks):
    """
    Perform inference on a prompt.
    
    Handles:
    - Concurrency limits
    - Queue backpressure
    - Timeouts
    - Metrics collection
    """
    global inference_engine, metrics_collector, request_semaphore
    
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Inference engine not ready")
    
    # Check queue size (backpressure)
    queue_size = inference_engine.batcher.get_queue_size()
    if queue_size >= settings.max_queue_size:
        raise HTTPException(
            status_code=503,
            detail=f"Queue full: {queue_size}/{settings.max_queue_size}"
        )
    
    # Acquire semaphore (concurrency limit)
    async with request_semaphore:
        request_id = request.request_id or str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Perform inference
            result = await inference_engine.infer(
                request_id=request_id,
                input_text=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                use_cache=request.use_cache
            )
            
            latency = (time.time() - start_time) * 1000  # Convert to ms
            
            # Record metrics
            if metrics_collector:
                tokens = len(result.split())  # Approximate token count
                background_tasks.add_task(
                    metrics_collector.record_request,
                    latency / 1000.0,  # Convert to seconds
                    tokens,
                    False
                )
                background_tasks.add_task(
                    metrics_collector.update_queue_size,
                    queue_size
                )
            
            return InferenceResponse(
                text=result,
                request_id=request_id,
                latency_ms=latency,
                tokens_generated=len(result.split()),
                cached=False  # TODO: Track cache hits
            )
        
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            if metrics_collector:
                background_tasks.add_task(
                    metrics_collector.record_request,
                    latency / 1000.0,
                    0,
                    True
                )
            logger.error(f"Inference error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    global inference_engine, metrics_collector
    
    model_loaded = inference_engine is not None and inference_engine.runtime.model is not None
    queue_size = inference_engine.batcher.get_queue_size() if inference_engine else 0
    
    metrics = {}
    if metrics_collector:
        metrics = metrics_collector.get_all_metrics()
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        queue_size=queue_size,
        metrics=metrics
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get performance metrics."""
    global metrics_collector, inference_engine
    
    if not metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics not available")
    
    metrics = metrics_collector.get_all_metrics()
    
    # Add engine stats
    if inference_engine:
        metrics["engine"] = inference_engine.get_stats()
    
    return MetricsResponse(metrics=metrics)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "LLM Inference Engine",
        "version": "1.0.0",
        "docs": "/docs"
    }
