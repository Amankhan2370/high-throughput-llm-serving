"""
Dynamic batching system for efficient inference.
Implements batching with timeout and size limits.
"""
import asyncio
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """Individual request in a batch."""
    request_id: str
    input_text: str
    max_tokens: int
    temperature: float
    top_p: float
    future: asyncio.Future
    timestamp: float


class DynamicBatcher:
    """
    Dynamic batching system that groups requests for efficient processing.
    """
    
    def __init__(
        self,
        max_batch_size: int = 32,
        batch_timeout_ms: int = 50,
        max_sequence_length: int = 2048
    ):
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms / 1000.0  # Convert to seconds
        self.max_sequence_length = max_sequence_length
        
        self.queue: deque = deque()
        self.batch_processor: Optional[Callable] = None
        self._running = False
        self._lock = asyncio.Lock()
    
    def set_processor(self, processor: Callable):
        """Set the batch processing function."""
        self.batch_processor = processor
    
    async def add_request(
        self,
        request_id: str,
        input_text: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> asyncio.Future:
        """
        Add a request to the batch queue.
        
        Returns:
            Future that will be resolved with the result
        """
        future = asyncio.Future()
        
        request = BatchRequest(
            request_id=request_id,
            input_text=input_text,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            future=future,
            timestamp=time.time()
        )
        
        async with self._lock:
            self.queue.append(request)
        
        # Trigger batch processing if queue is full
        if len(self.queue) >= self.max_batch_size:
            asyncio.create_task(self._process_batch())
        
        return future
    
    async def start(self):
        """Start the background batch processor."""
        if self._running:
            return
        
        self._running = True
        asyncio.create_task(self._batch_loop())
    
    async def stop(self):
        """Stop the batch processor."""
        self._running = False
        # Process remaining requests
        while self.queue:
            await self._process_batch()
    
    async def _batch_loop(self):
        """Background loop that processes batches on timeout."""
        while self._running:
            await asyncio.sleep(self.batch_timeout_ms)
            
            async with self._lock:
                if self.queue:
                    await self._process_batch()
    
    async def _process_batch(self):
        """Process a batch of requests."""
        if not self.batch_processor:
            logger.error("No batch processor set")
            return
        
        async with self._lock:
            if not self.queue:
                return
            
            # Collect requests for batch
            batch_requests: List[BatchRequest] = []
            batch_size = min(len(self.queue), self.max_batch_size)
            
            for _ in range(batch_size):
                if self.queue:
                    batch_requests.append(self.queue.popleft())
        
        if not batch_requests:
            return
        
        try:
            # Prepare batch inputs
            batch_inputs = [req.input_text for req in batch_requests]
            batch_configs = [
                {
                    "max_tokens": req.max_tokens,
                    "temperature": req.temperature,
                    "top_p": req.top_p
                }
                for req in batch_requests
            ]
            
            # Process batch
            results = await self.batch_processor(batch_inputs, batch_configs)
            
            # Resolve futures
            for i, request in enumerate(batch_requests):
                if i < len(results):
                    request.future.set_result(results[i])
                else:
                    request.future.set_exception(
                        Exception(f"Result missing for request {request.request_id}")
                    )
        
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            # Set exception for all requests in batch
            for request in batch_requests:
                if not request.future.done():
                    request.future.set_exception(e)
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return len(self.queue)
