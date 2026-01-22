"""
Metrics collection for QPS, latency, and system metrics.
Supports Prometheus integration.
"""
import time
import psutil
from typing import Dict, Any, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and tracks performance metrics."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        
        # Latency tracking
        self.latency_history: deque = deque(maxlen=window_size)
        self.request_times: deque = deque(maxlen=window_size)
        
        # Counters
        self.total_requests = 0
        self.total_errors = 0
        self.total_tokens_generated = 0
        
        # Prometheus metrics (if enabled)
        self.prometheus_enabled = False
        self._init_prometheus()
    
    def _init_prometheus(self):
        """Initialize Prometheus metrics if available."""
        try:
            from prometheus_client import Counter, Histogram, Gauge, start_http_server
            self.prometheus_enabled = True
            
            self.request_counter = Counter(
                'llm_requests_total',
                'Total number of requests',
                ['status']
            )
            
            self.latency_histogram = Histogram(
                'llm_request_latency_seconds',
                'Request latency in seconds',
                buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            )
            
            self.queue_size_gauge = Gauge(
                'llm_queue_size',
                'Current queue size'
            )
            
            self.tokens_gauge = Gauge(
                'llm_tokens_generated_total',
                'Total tokens generated'
            )
            
            logger.info("Prometheus metrics initialized")
        except ImportError:
            logger.warning("Prometheus client not available")
            self.prometheus_enabled = False
    
    def start_prometheus_server(self, port: int = 9090):
        """Start Prometheus metrics server."""
        if not self.prometheus_enabled:
            return
        
        try:
            from prometheus_client import start_http_server
            start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {str(e)}")
    
    def record_request(self, latency: float, tokens: int = 0, error: bool = False):
        """Record a request with latency and tokens."""
        self.total_requests += 1
        self.latency_history.append(latency)
        self.request_times.append(time.time())
        
        if tokens > 0:
            self.total_tokens_generated += tokens
        
        if error:
            self.total_errors += 1
        
        if self.prometheus_enabled:
            status = "error" if error else "success"
            self.request_counter.labels(status=status).inc()
            self.latency_histogram.observe(latency)
            if tokens > 0:
                self.tokens_gauge.inc(tokens)
    
    def update_queue_size(self, size: int):
        """Update queue size metric."""
        if self.prometheus_enabled:
            self.queue_size_gauge.set(size)
    
    def get_qps(self) -> float:
        """Calculate current QPS (requests per second)."""
        if len(self.request_times) < 2:
            return 0.0
        
        time_window = self.request_times[-1] - self.request_times[0]
        if time_window == 0:
            return 0.0
        
        return len(self.request_times) / time_window
    
    def get_latency_percentiles(self) -> Dict[str, float]:
        """Get latency percentiles (p50, p95, p99)."""
        if not self.latency_history:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        
        sorted_latencies = sorted(self.latency_history)
        n = len(sorted_latencies)
        
        return {
            "p50": sorted_latencies[int(n * 0.50)],
            "p95": sorted_latencies[int(n * 0.95)],
            "p99": sorted_latencies[int(n * 0.99)],
            "min": sorted_latencies[0],
            "max": sorted_latencies[-1],
            "mean": sum(sorted_latencies) / n
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system resource metrics."""
        process = psutil.Process()
        
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_mb": process.memory_info().rss / 1024 / 1024,
            "threads": process.num_threads()
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            "requests": {
                "total": self.total_requests,
                "errors": self.total_errors,
                "success_rate": 1.0 - (self.total_errors / max(self.total_requests, 1))
            },
            "throughput": {
                "qps": self.get_qps(),
                "tokens_generated": self.total_tokens_generated
            },
            "latency": self.get_latency_percentiles(),
            "system": self.get_system_metrics()
        }
