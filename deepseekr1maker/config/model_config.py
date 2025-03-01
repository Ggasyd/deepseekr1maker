from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Tuple
#import os
import torch

@dataclass
class TokenizerConfig:
    """Configuration for the tokenizer."""
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer name or path. If None, uses model_name_or_path."}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Use fast tokenizer implementation if available."}
    )
    padding_side: str = field(
        default="right",
        metadata={"help": "Padding side: 'right' or 'left'."}
    )
    truncation_side: str = field(
        default="right",
        metadata={"help": "Truncation side: 'right' or 'left'."}
    )
    add_eos_token: bool = field(
        default=True,
        metadata={"help": "Add EOS token to the end of the sequence."}
    )
    add_bos_token: bool = field(
        default=True,
        metadata={"help": "Add BOS token to the beginning of the sequence."}
    )
    special_tokens: Dict[str, str] = field(
        default_factory=lambda: {},
        metadata={"help": "Additional special tokens to add to the tokenizer."}
    )


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = field(
        default=0.7,
        metadata={"help": "Sampling temperature for generation."}
    )
    top_p: float = field(
        default=0.9,
        metadata={"help": "Top-p (nucleus) sampling parameter."}
    )
    top_k: int = field(
        default=50,
        metadata={"help": "Top-k sampling parameter."}
    )
    num_beams: int = field(
        default=1,
        metadata={"help": "Number of beams for beam search. 1 means no beam search."}
    )
    max_new_tokens: int = field(
        default=1024,
        metadata={"help": "Maximum number of new tokens to generate."}
    )
    min_new_tokens: int = field(
        default=0,
        metadata={"help": "Minimum number of new tokens to generate."}
    )
    repetition_penalty: float = field(
        default=1.1,
        metadata={"help": "Repetition penalty for generation."}
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={"help": "Presence penalty for generation."}
    )
    frequency_penalty: float = field(
        default=0.0,
        metadata={"help": "Frequency penalty for generation."}
    )
    length_penalty: float = field(
        default=1.0,
        metadata={"help": "Length penalty for beam search."}
    )
    do_sample: bool = field(
        default=True,
        metadata={"help": "Whether to use sampling; use greedy decoding otherwise."}
    )
    use_cache: bool = field(
        default=True,
        metadata={"help": "Use KV cache during generation."}
    )
    early_stopping: bool = field(
        default=True,
        metadata={"help": "Stop beam search when num_beams hypotheses are finished."}
    )
    

@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    quantization_method: Optional[str] = field(
        default=None,
        metadata={"help": "Quantization method: None, 'bitsandbytes-4bit', 'bitsandbytes-8bit', 'gptq', 'awq'"}
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load model in 4-bit precision using bitsandbytes."}
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load model in 8-bit precision using bitsandbytes."}
    )
    bnb_4bit_compute_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for 4-bit quantization: float16, bfloat16, float32"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type for 4-bit bitsandbytes: fp4 or nf4"}
    )
    bnb_4bit_use_double_quant: bool = field(
        default=True,
        metadata={"help": "Use nested quantization for 4-bit bitsandbytes"}
    )


@dataclass
class PeftConfig:
    """Configuration for Parameter-Efficient Fine-Tuning (PEFT)."""
    peft_method: Optional[str] = field(
        default=None,
        metadata={"help": "PEFT method: None, 'lora', 'qlora', 'adalora', 'prefix_tuning', 'p_tuning', 'prompt_tuning'"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank parameter."}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha parameter."}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout rate."}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "List of module names to apply LoRA to."}
    )
    prefix_tuning_length: int = field(
        default=30,
        metadata={"help": "Length of prefix tuning tokens."}
    )
    peft_task_type: str = field(
        default="CAUSAL_LM",
        metadata={"help": "Task type for PEFT: CAUSAL_LM, SEQ_CLS, SEQ_2_SEQ_LM, TOKEN_CLS"}
    )


@dataclass
class ModelParallelConfig:
    """Configuration for model parallelism and distribution."""
    device_map: Union[str, Dict[str, int]] = field(
        default="auto",
        metadata={"help": "Device map for model distribution: 'auto', 'balanced', 'balanced_low_0', or custom dict."}
    )
    max_memory_per_gpu: Optional[Dict[int, str]] = field(
        default=None,
        metadata={"help": "Maximum memory per GPU in format {0: '10GiB', 1: '10GiB'}."}
    )
    offload_folder: Optional[str] = field(
        default=None,
        metadata={"help": "Offload folder path for CPU offloading."}
    )
    offload_state_dict: bool = field(
        default=False,
        metadata={"help": "Offload state dict to CPU to save GPU memory."}
    )


@dataclass
class ModelConfig:
    """Enhanced configuration for the DeepSeek R1 model."""
    # Base model settings
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-0.5B-Instruct", 
        metadata={"help": "Path to pretrained model or identifier from huggingface.co/models"}
    )
    model_revision: Optional[str] = field(
        default="main", 
        metadata={"help": "Specific model version to use (branch, tag or commit ID)."}
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16", 
        metadata={"help": "Torch data type to use (float16, bfloat16, float32)."}
    )
    trust_remote_code: bool = field(
        default=True, 
        metadata={"help": "Allow remote code when loading model and tokenizer."}
    )
    attn_implementation: Optional[str] = field(
        default="flash_attention_2", 
        metadata={"help": "Attention implementation to use: 'flash_attention_2', 'sdpa', 'eager', or None."}
    )
    max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length for the model."}
    )
    
    # Model family specifiers
    model_family: Optional[str] = field(
        default=None,
        metadata={"help": "Model family (llama, qwen, mistral, etc.) for auto-configuration."}
    )
    model_variant: str = field(
        default="instruct",
        metadata={"help": "Model variant: 'base', 'instruct', 'chat', etc."}
    )
    
    # Advanced model configuration 
    model_config_overrides: Dict[str, Any] = field(
        default_factory=lambda: {},
        metadata={"help": "Override specific parameters in the model configuration."}
    )
    rope_scaling: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "RoPE scaling configuration for extended context, e.g. {'type': 'dynamic', 'factor': 2.0}"}
    )
    use_sliding_window: bool = field(
        default=False,
        metadata={"help": "Use sliding window attention for handling long sequences."}
    )
    sliding_window: Optional[int] = field(
        default=None,
        metadata={"help": "Size of sliding window for attention."}
    )
    use_flash_attn_2: bool = field(
        default=True,
        metadata={"help": "Use Flash Attention 2 for faster training and inference."}
    )
    
    # Additional components
    tokenizer: TokenizerConfig = field(
        default_factory=TokenizerConfig,
        metadata={"help": "Tokenizer configuration."}
    )
    generation: GenerationConfig = field(
        default_factory=GenerationConfig,
        metadata={"help": "Generation configuration."}
    )
    quantization: QuantizationConfig = field(
        default_factory=QuantizationConfig,
        metadata={"help": "Quantization configuration."}
    )
    peft: PeftConfig = field(
        default_factory=PeftConfig,
        metadata={"help": "PEFT configuration."}
    )
    model_parallel: ModelParallelConfig = field(
        default_factory=ModelParallelConfig,
        metadata={"help": "Model parallelism configuration."}
    )
    
    # Training-specific model settings
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Use gradient checkpointing to save memory."}
    )
    use_cache: bool = field(
        default=True,
        metadata={"help": "Use KV cache during generation."}
    )
    
    # LLM-specific settings
    prompt_template: Optional[str] = field(
        default=None,
        metadata={"help": "Prompt template for the model, e.g. '<|im_start|>user\\n{input}<|im_end|>\\n<|im_start|>assistant\\n'"}
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "System prompt to use for the model."}
    )
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Set quantization method based on load_in_Xbit settings
        if self.quantization.load_in_4bit and not self.quantization.quantization_method:
            self.quantization.quantization_method = "bitsandbytes-4bit"
        elif self.quantization.load_in_8bit and not self.quantization.quantization_method:
            self.quantization.quantization_method = "bitsandbytes-8bit"
            
        # Set proper dtype based on quantization
        if self.quantization.quantization_method in ["bitsandbytes-4bit", "bitsandbytes-8bit"]:
            if self.torch_dtype == "bfloat16":
                self.quantization.bnb_4bit_compute_dtype = "bfloat16"
        
        # Set tokenizer path if not specified
        if not self.tokenizer.tokenizer_name_or_path:
            self.tokenizer.tokenizer_name_or_path = self.model_name_or_path
        
        # Set attention implementation based on flash_attn_2 flag
        if self.use_flash_attn_2 and self.attn_implementation is None:
            self.attn_implementation = "flash_attention_2"
        
        # Set sliding window if specified
        if self.use_sliding_window and self.sliding_window is None:
            self.sliding_window = 4096  # Default sliding window size
            
    def get_peft_config(self) -> Dict[str, Any]:
        """Get PEFT configuration for HuggingFace PEFT library."""
        if self.peft.peft_method == "lora" or self.peft.peft_method == "qlora":
            return {
                "r": self.peft.lora_r,
                "lora_alpha": self.peft.lora_alpha,
                "lora_dropout": self.peft.lora_dropout,
                "bias": "none",
                "task_type": self.peft.peft_task_type,
                "target_modules": self.peft.lora_target_modules,
            }
        elif self.peft.peft_method == "prefix_tuning":
            return {
                "num_virtual_tokens": self.peft.prefix_tuning_length,
                "task_type": self.peft.peft_task_type,
            }
        return {}
    
    def get_model_kwargs(self) -> Dict[str, Any]:
        """Get model initialization kwargs for transformer models."""
        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": getattr(torch, self.torch_dtype) if hasattr(torch, self.torch_dtype) else None,
            "use_cache": self.use_cache,
        }
        
        # Add quantization parameters
        if self.quantization.quantization_method == "bitsandbytes-4bit":
            model_kwargs.update({
                "load_in_4bit": True,
                "quantization_config": {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": getattr(torch, self.quantization.bnb_4bit_compute_dtype),
                    "bnb_4bit_quant_type": self.quantization.bnb_4bit_quant_type,
                    "bnb_4bit_use_double_quant": self.quantization.bnb_4bit_use_double_quant,
                }
            })
        elif self.quantization.quantization_method == "bitsandbytes-8bit":
            model_kwargs.update({
                "load_in_8bit": True,
                "quantization_config": {
                    "load_in_8bit": True,
                }
            })
            
        # Add model parallel parameters
        if self.model_parallel.device_map is not None:
            model_kwargs["device_map"] = self.model_parallel.device_map
            
        if self.model_parallel.max_memory_per_gpu is not None:
            model_kwargs["max_memory"] = self.model_parallel.max_memory_per_gpu
            
        if self.model_parallel.offload_folder is not None:
            model_kwargs["offload_folder"] = self.model_parallel.offload_folder
            
        if self.model_parallel.offload_state_dict:
            model_kwargs["offload_state_dict"] = True
            
        # Add attention implementation
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation
            
        # Add RoPE scaling for extended context
        if self.rope_scaling:
            model_kwargs["rope_scaling"] = self.rope_scaling
            
        # Add sliding window attention
        if self.use_sliding_window and self.sliding_window:
            model_kwargs["sliding_window"] = self.sliding_window
            
        # Add gradient checkpointing
        if self.gradient_checkpointing:
            model_kwargs["gradient_checkpointing"] = True
            
        # Add any additional model config overrides
        if self.model_config_overrides:
            model_kwargs.update(self.model_config_overrides)
            
        return model_kwargs