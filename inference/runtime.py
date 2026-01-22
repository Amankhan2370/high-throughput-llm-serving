"""
Inference runtime for executing model inference.
Handles model loading, device management, and forward passes.
"""
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class InferenceRuntime:
    """Core inference runtime for LLM models."""
    
    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        device: str = "cuda",
        max_sequence_length: int = 2048,
        use_fp16: bool = False,
        use_bf16: bool = False,
        torch_compile: bool = False
    ):
        self.model_name = model_name
        self.model_path = model_path or model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_sequence_length = max_sequence_length
        self.use_fp16 = use_fp16 and self.device == "cuda"
        self.use_bf16 = use_bf16 and self.device == "cuda"
        self.torch_compile = torch_compile
        
        self.model: Optional[nn.Module] = None
        self.tokenizer: Optional[Any] = None
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_path} on device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.use_fp16 else (torch.bfloat16 if self.use_bf16 else torch.float32),
                device_map=self.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            self.model.eval()
            
            # Compile model if requested (PyTorch 2.0+)
            if self.torch_compile and hasattr(torch, 'compile'):
                logger.info("Compiling model with torch.compile")
                self.model = torch.compile(self.model)
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize input texts."""
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        Generate tokens from input.
        
        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Generated token IDs
        """
        with torch.no_grad():
            # Move to device
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            # Generate
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            return outputs
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """Decode token IDs to text."""
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self.model is None:
            return {}
        
        num_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.model_name,
            "device": self.device,
            "total_parameters": num_params,
            "trainable_parameters": trainable_params,
            "dtype": str(next(self.model.parameters()).dtype),
            "max_sequence_length": self.max_sequence_length
        }
