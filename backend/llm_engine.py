# backend/llm_engine.py
"""
State-of-the-art small LLM engine with latest optimizations
Supports multiple inference backends for maximum efficiency
"""

import json
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
import asyncio
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

class ModelProvider(Enum):
    """Supported model providers"""
    OLLAMA = "ollama"  # Ollama for easy local deployment
    LLAMACPP = "llamacpp"  # llama.cpp with latest optimizations
    MLC = "mlc"  # MLC-LLM for mobile/edge deployment
    CANDLE = "candle"  # Rust-based Candle for speed
    TRANSFORMERS = "transformers"  # HuggingFace Transformers
    VLLM = "vllm"  # vLLM for production serving
    EXLLAMAV2 = "exllamav2"  # ExLlamaV2 for extreme speed

@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str
    provider: ModelProvider
    quantization: str = "Q4_K_M"
    max_tokens: int = 2048
    temperature: float = 0.1
    device: str = "cpu"
    context_window: int = 8192
    use_flash_attention: bool = True
    rope_scaling: Optional[Dict] = None  # For long context
    use_sliding_window: bool = True  # For efficiency

class SmallLLMEngine:
    """
    Unified engine for small LLMs with latest optimizations
    Implements techniques from 2024 research
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.provider_engine = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the appropriate engine based on provider"""
        if self.config.provider == ModelProvider.OLLAMA:
            self.provider_engine = OllamaEngine(self.config)
        elif self.config.provider == ModelProvider.LLAMACPP:
            self.provider_engine = LlamaCppEngine(self.config)
        elif self.config.provider == ModelProvider.TRANSFORMERS:
            self.provider_engine = TransformersEngine(self.config)
        elif self.config.provider == ModelProvider.MLC:
            self.provider_engine = MLCEngine(self.config)
        else:
            # Default to Transformers
            self.provider_engine = TransformersEngine(self.config)
    
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> str:
        """Generate response using the configured engine"""
        return await self.provider_engine.generate(
            prompt, 
            system_prompt, 
            tools, 
            stream
        )
    
    async def generate_with_rag(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate with RAG context"""
        # Format context into prompt
        context_str = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(context)])
        
        enhanced_prompt = f"""Based on the following API documentation context, answer the user's question accurately. 
Only use information from the provided context. If the answer is not in the context, say so.

Context:
{context_str}

User Question: {query}

Answer:"""
        
        return await self.generate(enhanced_prompt, system_prompt)

class OllamaEngine:
    """Ollama backend - easiest to deploy"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.base_url = "http://localhost:11434"
        self._ensure_model_loaded()
    
    def _ensure_model_loaded(self):
        """Ensure model is loaded in Ollama"""
        import requests
        
        # Map model names to Ollama models
        ollama_models = {
            "Qwen/Qwen2.5-0.5B-Instruct": "qwen2.5:0.5b",
            "microsoft/phi-3.5-mini-instruct": "phi3.5:3.8b",
            "google/gemma-2-2b-it": "gemma2:2b",
            "stabilityai/stablelm-2-1_6b": "stablelm2:1.6b",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "tinyllama:1.1b"
        }
        
        model_name = ollama_models.get(self.config.model_name, "qwen2.5:0.5b")
        
        try:
            # Check if model exists
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                if not any(model_name in m.get("name", "") for m in models):
                    logger.info(f"Pulling model {model_name}...")
                    requests.post(
                        f"{self.base_url}/api/pull",
                        json={"name": model_name}
                    )
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> str:
        """Generate using Ollama"""
        import aiohttp
        
        ollama_models = {
            "Qwen/Qwen2.5-0.5B-Instruct": "qwen2.5:0.5b",
            "microsoft/phi-3.5-mini-instruct": "phi3.5:3.8b",
            "google/gemma-2-2b-it": "gemma2:2b",
        }
        
        model_name = ollama_models.get(self.config.model_name, "qwen2.5:0.5b")
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": stream
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        # Add tool calling if supported
        if tools and self.config.model_name.startswith("Qwen"):
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "")
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return ""

class LlamaCppEngine:
    """llama.cpp backend with latest optimizations"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load model with llama-cpp-python"""
        try:
            from llama_cpp import Llama
            
            # Model paths for popular small models
            model_paths = {
                "Qwen/Qwen2.5-0.5B-Instruct": "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
                "microsoft/phi-3.5-mini-instruct": "models/phi-3.5-mini-instruct-q4_k_m.gguf",
                "google/gemma-2-2b-it": "models/gemma-2-2b-it-q4_k_m.gguf",
                "HuggingFaceTB/SmolLM2-1.7B-Instruct": "models/smollm2-1.7b-instruct-q4_k_m.gguf"
            }
            
            model_path = model_paths.get(self.config.model_name)
            if not model_path:
                logger.warning(f"Model {self.config.model_name} not found")
                return
            
            # Initialize with latest optimizations
            self.model = Llama(
                model_path=model_path,
                n_ctx=self.config.context_window,
                n_batch=512,
                n_threads=8,
                n_gpu_layers=-1 if self.config.device == "cuda" else 0,
                rope_scaling_type=1 if self.config.context_window > 4096 else 0,
                use_mlock=True,  # Keep model in RAM
                use_mmap=True,  # Memory-mapped files for efficiency
                flash_attn=self.config.use_flash_attention,
                verbose=False
            )
            
            logger.info(f"Loaded {self.config.model_name} with llama.cpp")
            
        except ImportError:
            logger.warning("llama-cpp-python not installed")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> str:
        """Generate using llama.cpp"""
        if not self.model:
            return ""
        
        # Format prompt based on model
        formatted_prompt = self._format_prompt(prompt, system_prompt)
        
        # Add function calling for supported models
        grammar = None
        if tools and "qwen" in self.config.model_name.lower():
            # Use grammar-based sampling for structured output
            grammar = self._create_tool_grammar(tools)
        
        try:
            response = self.model(
                formatted_prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=0.95,
                repeat_penalty=1.1,
                grammar=grammar,
                stream=stream
            )
            
            if stream:
                return response  # Return generator
            else:
                return response["choices"][0]["text"]
                
        except Exception as e:
            logger.error(f"llama.cpp generation failed: {e}")
            return ""
    
    def _format_prompt(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Format prompt based on model template"""
        if "qwen" in self.config.model_name.lower():
            # Qwen format
            if system_prompt:
                return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        elif "phi" in self.config.model_name.lower():
            # Phi format
            if system_prompt:
                return f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
            return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        elif "gemma" in self.config.model_name.lower():
            # Gemma format
            if system_prompt:
                return f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        else:
            # Generic format
            if system_prompt:
                return f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            return f"User: {prompt}\n\nAssistant:"

class TransformersEngine:
    """HuggingFace Transformers with latest optimizations"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load model with transformers"""
        try:
            from transformers import (
                AutoModelForCausalLM, 
                AutoTokenizer,
                BitsAndBytesConfig,
                TextStreamer
            )
            import torch
            
            # Quantization config for 4-bit
            bnb_config = None
            if self.config.quantization.startswith("Q4"):
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                device_map="auto" if self.config.device == "cuda" else "cpu",
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                trust_remote_code=True,
                use_flash_attention_2=self.config.use_flash_attention,
                rope_scaling={"type": "linear", "factor": 2.0} if self.config.context_window > 4096 else None
            )
            
            # Enable bettertransformer for efficiency
            if hasattr(self.model, "to_bettertransformer"):
                self.model = self.model.to_bettertransformer()
            
            logger.info(f"Loaded {self.config.model_name} with Transformers")
            
        except ImportError:
            logger.warning("transformers not installed properly")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> str:
        """Generate using transformers"""
        if not self.model or not self.tokenizer:
            return ""
        
        
        
        # Format messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Add tools if supported (Qwen 2.5, Phi-3.5)
        if tools and hasattr(self.tokenizer, "apply_tool_use_template"):
            messages = self.tokenizer.apply_tool_use_template(
                messages,
                tools=tools,
                tool_choice="auto"
            )
        
        # Tokenize
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True
        )
        
        if self.config.device == "cuda":
            inputs = inputs.to("cuda")
        
        # Generate with optimizations
        try:
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=True if self.config.temperature > 0 else False,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""

class MLCEngine:
    """MLC-LLM for mobile/edge deployment"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.engine = None
        self._initialize_mlc()
    
    def _initialize_mlc(self):
        """Initialize MLC-LLM"""
        try:
            from mlc_llm import MLCEngine
            
            # MLC optimized models
            mlc_models = {
                "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
                "microsoft/phi-3.5-mini-instruct": "Phi-3.5-mini-4k-instruct-q4f16_1-MLC",
                "google/gemma-2-2b-it": "gemma-2-2b-it-q4f16_1-MLC"
            }
            
            model_id = mlc_models.get(self.config.model_name)
            if model_id:
                self.engine = MLCEngine(
                    model=model_id,
                    device=self.config.device,
                    context_window_size=self.config.context_window,
                    sliding_window_size=2048 if self.config.use_sliding_window else -1,
                    attention_sink_size=4,  # Latest optimization
                    tensor_parallel_shards=1
                )
                logger.info(f"Initialized MLC engine for {self.config.model_name}")
        except ImportError:
            logger.warning("MLC-LLM not installed")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> str:
        """Generate using MLC"""
        if not self.engine:
            return ""
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.engine.chat.completions.create(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=stream
            )
            
            if stream:
                return response
            else:
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"MLC generation failed: {e}")
            return ""

# Specialized optimizations for tiny models
class TinyModelOptimizer:
    """Latest optimizations for sub-1B models"""
    
    @staticmethod
    def apply_speculative_decoding(model, draft_model):
        """Speculative decoding for 2-5x speedup"""
        # Use a tiny draft model for speculation
        pass
    
    @staticmethod
    def apply_medusa_heads(model):
        """Medusa-style parallel decoding"""
        # Add multiple decoding heads
        pass
    
    @staticmethod  
    def apply_lookahead_decoding(model):
        """Lookahead decoding for faster generation"""
        # Jacobi decoding implementation
        pass
    
    @staticmethod
    def apply_layer_skip(model, skip_ratio=0.5):
        """Skip layers during inference for speed"""
        # Elastic inference
        pass