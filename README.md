# LLM Inference Engine

Production-ready LLM inference and serving system with dynamic batching, caching, and high-throughput capabilities.

## Overview

This system provides a complete, end-to-end LLM inference engine designed for production workloads. It handles high QPS with low latency through dynamic batching, implements intelligent caching, supports async request handling, and includes comprehensive observability.

**Key Features:**
- Dynamic batching with configurable timeouts
- In-memory and Redis caching support
- Async request handling with concurrency limits
- Request queuing with backpressure
- Comprehensive metrics (QPS, p50, p95, p99 latency)
- Prometheus integration
- Timeout and error handling
- Production-grade structure

## Architecture

```
┌─────────────┐
│  FastAPI    │
│   Server    │
└──────┬──────┘
       │
┌──────▼──────────┐
│ Request Queue   │
│ (Backpressure)  │
└──────┬──────────┘
       │
┌──────▼──────────┐
│ Dynamic Batcher │
│ (Timeout-based) │
└──────┬──────────┘
       │
┌──────▼──────────┐
│  Cache Layer    │
│ (Memory/Redis)  │
└──────┬──────────┘
       │
┌──────▼──────────┐
│ Inference       │
│ Runtime         │
│ (PyTorch)       │
└─────────────────┘
```

### Components

1. **FastAPI Server**: HTTP API with async handling, timeouts, and concurrency limits
2. **Dynamic Batcher**: Groups requests for efficient batch processing
3. **Cache Layer**: In-memory or Redis-backed caching for repeated queries
4. **Inference Runtime**: PyTorch-based model loading and execution
5. **Metrics Collector**: Tracks QPS, latency percentiles, and system metrics

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- Model files (HuggingFace format)
- Redis (optional, for distributed caching)

### Installation

```bash
# Clone repository
git clone https://github.com/Amankhan2370/llm-inference-engine.git
cd llm-inference-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your configuration
```

### Environment Variables

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `MODEL_NAME` | HuggingFace model name | Yes | `gpt2` |
| `MODEL_PATH` | Local model path (optional) | No | `/path/to/model` |
| `DEVICE` | Device (cuda/cpu) | Yes | `cuda` |
| `REDIS_URL` | Redis connection URL | No* | `redis://localhost:6379` |
| `CACHE_TYPE` | Cache backend (memory/redis) | Yes | `memory` |
| `MAX_BATCH_SIZE` | Maximum batch size | Yes | `32` |
| `BATCH_TIMEOUT_MS` | Batch timeout in ms | Yes | `50` |
| `MAX_CONCURRENT_REQUESTS` | Concurrency limit | Yes | `100` |
| `MAX_QUEUE_SIZE` | Queue size limit | Yes | `1000` |

*Required if `CACHE_TYPE=redis`

**Important**: All sensitive values must be provided via environment variables. Never commit real credentials.

## Running

### Local Development

```bash
# Using script
./scripts/run_local.sh

# Or directly
python main.py
```

### Docker

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f
```

### Production

```bash
# Using uvicorn with workers
uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 1
```

**Note**: Use single worker for shared model instance. For multiple workers, implement model sharding.

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "queue_size": 0,
  "metrics": {...}
}
```

### Inference Request

```bash
curl -X POST http://localhost:8000/api/v1/infer \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

**Response:**
```json
{
  "text": "The future of AI is bright and full of possibilities...",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "latency_ms": 245.3,
  "tokens_generated": 42,
  "cached": false
}
```

### Metrics

```bash
curl http://localhost:8000/metrics
```

**Response includes:**
- Request counts and error rates
- QPS (queries per second)
- Latency percentiles (p50, p95, p99)
- System metrics (CPU, memory)
- Cache statistics

### Prometheus Metrics

If enabled, metrics are available at:
```
http://localhost:9090/metrics
```

## Performance Metrics

The system tracks:

- **QPS**: Queries per second
- **Latency**: p50, p95, p99 percentiles
- **Throughput**: Tokens generated per second
- **Cache Hit Rate**: Percentage of cached responses
- **Error Rate**: Failed requests percentage
- **Queue Size**: Current pending requests

## Failure Modes and Safeguards

### Timeout Protection
- Request timeout: 30s (configurable)
- Automatic timeout handling with 504 responses

### Backpressure
- Queue size limit: 1000 (configurable)
- 503 responses when queue is full
- Prevents memory exhaustion

### Concurrency Limits
- Max concurrent requests: 100 (configurable)
- Semaphore-based throttling
- Prevents resource exhaustion

### Error Handling
- Graceful error responses
- Metrics tracking for errors
- Logging for debugging

### Model Loading
- Automatic fallback to CPU if CUDA unavailable
- Error handling for missing models
- Health checks verify model status

## Project Structure

```
llm-inference-engine/
├── api/
│   └── server.py              # FastAPI application
├── inference/
│   ├── engine.py              # Main inference orchestrator
│   ├── runtime.py             # PyTorch model runtime
│   ├── batching.py            # Dynamic batching system
│   └── cache.py               # Caching layer
├── profiling/
│   └── metrics.py             # Metrics collection
├── config/
│   └── settings.py            # Configuration management
├── scripts/
│   └── run_local.sh           # Local run script
├── tests/
│   └── test_inference.py      # Test suite
├── main.py                    # Entry point
├── requirements.txt
├── .env.example
├── Dockerfile
└── README.md
```

## Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest tests/ --cov=inference --cov=api --cov=profiling
```

## Production Notes

### Scaling Considerations

1. **Single Model Instance**: Current implementation uses single model instance. For horizontal scaling:
   - Implement model sharding
   - Use load balancer
   - Consider model parallelism

2. **Caching**: Use Redis for distributed caching across instances

3. **Monitoring**: Enable Prometheus metrics and set up alerting

4. **Resource Limits**: Configure based on:
   - Model size
   - Expected QPS
   - Available GPU memory

### Performance Tuning

- **Batch Size**: Increase for higher throughput, decrease for lower latency
- **Batch Timeout**: Lower for faster responses, higher for better batching
- **Cache TTL**: Adjust based on query patterns
- **FP16/BF16**: Enable for faster inference on supported hardware

## License

Proprietary - All rights reserved

## Contact

For questions or support, contact the repository maintainer.
