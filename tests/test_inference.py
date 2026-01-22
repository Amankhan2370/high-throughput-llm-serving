"""
Tests for inference engine.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from inference.runtime import InferenceRuntime
from inference.batching import DynamicBatcher
from inference.cache import Cache
from inference.engine import InferenceEngine


@pytest.fixture
def mock_runtime():
    """Mock inference runtime."""
    runtime = Mock(spec=InferenceRuntime)
    runtime.tokenize = Mock(return_value={
        "input_ids": Mock(),
        "attention_mask": Mock()
    })
    runtime.generate = Mock(return_value=Mock())
    runtime.decode = Mock(return_value=["Generated text"])
    runtime.get_model_info = Mock(return_value={"model_name": "test-model"})
    return runtime


@pytest.fixture
def batcher():
    """Create batcher instance."""
    return DynamicBatcher(max_batch_size=4, batch_timeout_ms=100)


@pytest.fixture
def cache():
    """Create cache instance."""
    return Cache(cache_type="memory", ttl=3600, max_size=1000)


@pytest.mark.asyncio
async def test_batcher_add_request(batcher):
    """Test adding requests to batcher."""
    future = await batcher.add_request(
        request_id="test-1",
        input_text="Test prompt",
        max_tokens=50
    )
    assert future is not None
    assert batcher.get_queue_size() == 1


@pytest.mark.asyncio
async def test_cache_get_set(cache):
    """Test cache get and set operations."""
    config = {"max_tokens": 100}
    
    # Set cache
    await cache.set("test prompt", config, "cached result")
    
    # Get cache
    result = await cache.get("test prompt", config)
    assert result == "cached result"
    
    # Get non-existent
    result = await cache.get("other prompt", config)
    assert result is None


@pytest.mark.asyncio
async def test_inference_engine_infer(mock_runtime, batcher, cache):
    """Test inference engine inference."""
    engine = InferenceEngine(
        runtime=mock_runtime,
        batcher=batcher,
        cache=cache,
        cache_enabled=True
    )
    
    await engine.start()
    
    # Mock batch processor
    async def mock_processor(inputs, configs):
        return ["Generated text"] * len(inputs)
    
    batcher.set_processor(mock_processor)
    
    # Perform inference
    result = await engine.infer(
        request_id="test-1",
        input_text="Test prompt",
        max_tokens=50
    )
    
    assert result == "Generated text"
    
    await engine.stop()


def test_metrics_collector():
    """Test metrics collection."""
    from profiling.metrics import MetricsCollector
    
    collector = MetricsCollector()
    
    # Record requests
    collector.record_request(0.1, tokens=50)
    collector.record_request(0.2, tokens=100)
    collector.record_request(0.15, tokens=75)
    
    # Check metrics
    metrics = collector.get_all_metrics()
    assert metrics["requests"]["total"] == 3
    assert metrics["throughput"]["tokens_generated"] == 225
    
    # Check latency percentiles
    latency = collector.get_latency_percentiles()
    assert "p50" in latency
    assert "p95" in latency
    assert "p99" in latency
