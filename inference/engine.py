"""
Main inference engine that orchestrates runtime, batching, and caching.
"""
import asyncio
import time
from typing import List, Dict, Any, Optional
from inference.runtime import InferenceRuntime
from inference.batching import DynamicBatcher
from inference.cache import Cache
import logging

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Production-ready inference engine with batching, caching, and async support.
    """
    
    def __init__(
        self,
        runtime: InferenceRuntime,
        batcher: DynamicBatcher,
        cache: Optional[Cache] = None,
        cache_enabled: bool = True
    ):
        self.runtime = runtime
        self.batcher = batcher
        self.cache = cache
        self.cache_enabled = cache_enabled and cache is not None
        
        # Set batch processor
        self.batcher.set_processor(self._process_batch)
        
        # Metrics
        self.request_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    async def start(self):
        """Start the inference engine."""
        await self.batcher.start()
        logger.info("Inference engine started")
    
    async def stop(self):
        """Stop the inference engine."""
        await self.batcher.stop()
        logger.info("Inference engine stopped")
    
    async def infer(
        self,
        request_id: str,
        input_text: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_cache: bool = True
    ) -> str:
        """
        Perform inference on a single request.
        
        Args:
            request_id: Unique request identifier
            input_text: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            use_cache: Whether to use cache
            
        Returns:
            Generated text
        """
        self.request_count += 1
        
        # Check cache
        if self.cache_enabled and use_cache:
            config = {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            cached_result = await self.cache.get(input_text, config)
            if cached_result:
                self.cache_hits += 1
                return cached_result
        
        self.cache_misses += 1
        
        # Add to batch queue
        future = await self.batcher.add_request(
            request_id=request_id,
            input_text=input_text,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Wait for result
        result = await future
        return result
    
    async def _process_batch(
        self,
        batch_inputs: List[str],
        batch_configs: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Process a batch of inputs.
        
        Args:
            batch_inputs: List of input texts
            batch_configs: List of generation configs
            
        Returns:
            List of generated texts
        """
        start_time = time.time()
        
        try:
            # Tokenize
            tokenized = self.runtime.tokenize(batch_inputs)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            
            # Use first config for batch (in production, handle per-request configs)
            config = batch_configs[0] if batch_configs else {}
            
            # Generate
            output_ids = self.runtime.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.get("max_tokens", 100),
                temperature=config.get("temperature", 0.7),
                top_p=config.get("top_p", 0.9),
                do_sample=config.get("temperature", 0.7) > 0
            )
            
            # Decode
            # Extract only new tokens (remove input)
            generated_texts = []
            for i, output in enumerate(output_ids):
                input_len = input_ids[i].shape[0]
                new_tokens = output[input_len:]
                text = self.runtime.decode([new_tokens], skip_special_tokens=True)[0]
                generated_texts.append(text)
            
            # Cache results
            if self.cache_enabled and self.cache:
                for i, (input_text, config) in enumerate(zip(batch_inputs, batch_configs)):
                    if i < len(generated_texts):
                        await self.cache.set(input_text, config, generated_texts[i])
            
            elapsed = time.time() - start_time
            logger.info(f"Processed batch of {len(batch_inputs)} requests in {elapsed:.3f}s")
            
            return generated_texts
        
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "request_count": self.request_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / max(self.request_count, 1),
            "queue_size": self.batcher.get_queue_size(),
            "model_info": self.runtime.get_model_info()
        }
