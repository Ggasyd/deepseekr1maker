from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
import os

@dataclass
class TrainingConfig:
    """Enhanced configuration for DeepSeek R1 training pipeline."""
    # Basic output settings
    output_dir: str = field(
        default="./deepseek_r1_model",
        metadata={"help": "Output directory for checkpoints and logs"}
    )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={"help": "Overwrite the output directory if it exists"}
    )
    
    # Training stages
    stages: List[str] = field(
        default_factory=lambda: ["r1zero", "sft", "reasoning_rl", "rejection", "sft2", "distill"],
        metadata={"help": "Training stages to execute: 'r1zero', 'sft', 'reasoning_rl', 'rejection', 'sft2', 'distill'"}
    )
    
    # Training epochs and steps
    num_train_epochs: Dict[str, int] = field(
        default_factory=lambda: {"r1zero": 2, "sft": 3, "reasoning_rl": 2, "rejection": 1, "sft2": 2, "distill": 3},
        metadata={"help": "Number of training epochs for each stage"}
    )
    max_steps: Dict[str, int] = field(
        default_factory=lambda: {"r1zero": -1, "sft": -1, "reasoning_rl": -1, "rejection": -1, "sft2": -1, "distill": -1},
        metadata={"help": "Max steps for each stage. -1 means train for num_train_epochs."}
    )
    
    # Batch size configuration
    per_device_train_batch_size: Dict[str, int] = field(
        default_factory=lambda: {"r1zero": 4, "sft": 8, "reasoning_rl": 4, "rejection": 8, "sft2": 8, "distill": 16},
        metadata={"help": "Batch size per device during training for each stage"}
    )
    per_device_eval_batch_size: Dict[str, int] = field(
        default_factory=lambda: {"r1zero": 8, "sft": 16, "reasoning_rl": 8, "rejection": 16, "sft2": 16, "distill": 32},
        metadata={"help": "Batch size for evaluation for each stage"}
    )
    gradient_accumulation_steps: Dict[str, int] = field(
        default_factory=lambda: {"r1zero": 8, "sft": 4, "reasoning_rl": 16, "rejection": 4, "sft2": 4, "distill": 2},
        metadata={"help": "Number of steps to accumulate gradients for each stage"}
    )
    
    # Learning rate and optimization
    learning_rate: Dict[str, float] = field(
        default_factory=lambda: {"r1zero": 5e-5, "sft": 2e-5, "reasoning_rl": 1e-5, "rejection": 8e-6, "sft2": 1e-5, "distill": 3e-5},
        metadata={"help": "Initial learning rate for each stage"}
    )
    lr_scheduler_type: Dict[str, str] = field(
        default_factory=lambda: {"r1zero": "cosine", "sft": "cosine", "reasoning_rl": "cosine", "rejection": "linear", "sft2": "cosine", "distill": "cosine"},
        metadata={"help": "Learning rate scheduler type for each stage"}
    )
    warmup_ratio: Dict[str, float] = field(
        default_factory=lambda: {"r1zero": 0.1, "sft": 0.05, "reasoning_rl": 0.1, "rejection": 0.05, "sft2": 0.05, "distill": 0.05},
        metadata={"help": "Warmup ratio over training steps for each stage"}
    )
    weight_decay: Dict[str, float] = field(
        default_factory=lambda: {"r1zero": 0.01, "sft": 0.01, "reasoning_rl": 0.005, "rejection": 0.005, "sft2": 0.01, "distill": 0.01},
        metadata={"help": "Weight decay for each stage"}
    )
    
    # Optimization and precision
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm for gradient clipping"})
    fp16: bool = field(default=False, metadata={"help": "Use FP16 mixed precision"})
    bf16: bool = field(default=True, metadata={"help": "Use BF16 mixed precision"})
    
    # Early stopping
    early_stopping_patience: Dict[str, int] = field(
        default_factory=lambda: {"r1zero": 3, "sft": 5, "reasoning_rl": 3, "rejection": 2, "sft2": 3, "distill": 3},
        metadata={"help": "Number of evaluation checks with no improvement after which training will be stopped"}
    )
    early_stopping_threshold: float = field(
        default=0.001, 
        metadata={"help": "Minimum improvement required to count as improvement"}
    )
    
    # Evaluation and logging
    logging_steps: int = field(default=10, metadata={"help": "Logging frequency (in steps)"})
    save_steps: int = field(default=100, metadata={"help": "Saving frequency (in steps)"})
    eval_steps: int = field(default=100, metadata={"help": "Evaluation frequency (in steps)"})
    evaluation_strategy: str = field(
        default="steps", 
        metadata={"help": "Evaluation strategy ('no', 'steps', 'epoch')"}
    )
    save_strategy: str = field(
        default="steps", 
        metadata={"help": "Saving strategy ('no', 'steps', 'epoch')"}
    )
    save_total_limit: int = field(
        default=5, 
        metadata={"help": "Limit of the total number of checkpoints. Removes the oldest ones."}
    )
    
    # Distributed training
    local_rank: int = field(default=-1, metadata={"help": "Local rank for distributed training"})
    ddp_find_unused_parameters: bool = field(
        default=False, 
        metadata={"help": "Find unused parameters in DDP"}
    )
    ddp_bucket_cap_mb: int = field(
        default=25, 
        metadata={"help": "DDP bucket capacity in MB"}
    )
    deepspeed: Optional[str] = field(
        default=None, 
        metadata={"help": "Path to deepspeed config file"}
    )
    
    # FSDP settings
    fsdp: Optional[str] = field(
        default=None, 
        metadata={"help": "FSDP strategy, options: 'full_shard', 'shard_grad_op', 'hybrid_shard'"}
    )
    fsdp_min_num_params: int = field(
        default=1e6, 
        metadata={"help": "Minimum number of parameters for FSDP wrapping"}
    )
    
    # Device settings
    device: str = field(
        default="cuda", 
        metadata={"help": "Device to use: 'cuda', 'cpu', etc."}
    )
    
    # Reproducibility
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})
    
    # Memory optimization
    use_gradient_checkpointing: bool = field(
        default=True, 
        metadata={"help": "Use gradient checkpointing to save memory"}
    )
    use_cache: bool = field(
        default=True,
        metadata={"help": "Whether to use KV cache during generation"}
    )
    
    # Advanced options for each stage
    stage_options: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "r1zero": {"peft_lora_r": 8, "peft_lora_alpha": 16},
            "sft": {"peft_lora_r": 16, "peft_lora_alpha": 32},
            "reasoning_rl": {"reward_scaling": 0.1, "kl_coef": 0.05},
            "rejection": {"top_k": 4},
            "sft2": {"mixup_alpha": 0.2},
            "distill": {"temperature": 2.0}
        },
        metadata={"help": "Additional options specific to each training stage"}
    )
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure all stages have complete configurations
        all_stages = set(self.stages)
        for param_name in ['num_train_epochs', 'max_steps', 'learning_rate', 
                          'per_device_train_batch_size', 'per_device_eval_batch_size',
                          'gradient_accumulation_steps', 'warmup_ratio', 'weight_decay',
                          'early_stopping_patience', 'lr_scheduler_type']:
            param_dict = getattr(self, param_name)
            if not all(stage in param_dict for stage in all_stages):
                missing = all_stages - set(param_dict.keys())
                raise ValueError(f"Missing {param_name} configuration for stages: {missing}")
        
        # Check for DeepSpeed configuration
        if self.deepspeed is not None and not os.path.exists(self.deepspeed):
            raise ValueError(f"DeepSpeed config file not found at {self.deepspeed}")
            
    def get_stage_config(self, stage: str) -> Dict[str, Any]:
        """Get all configuration parameters for a specific stage."""
        if stage not in self.stages:
            raise ValueError(f"Invalid stage: {stage}. Available stages: {self.stages}")
            
        config = {
            "output_dir": os.path.join(self.output_dir, stage),
            "num_train_epochs": self.num_train_epochs[stage],
            "max_steps": self.max_steps[stage],
            "per_device_train_batch_size": self.per_device_train_batch_size[stage],
            "per_device_eval_batch_size": self.per_device_eval_batch_size[stage],
            "gradient_accumulation_steps": self.gradient_accumulation_steps[stage],
            "learning_rate": self.learning_rate[stage],
            "lr_scheduler_type": self.lr_scheduler_type[stage],
            "warmup_ratio": self.warmup_ratio[stage],
            "weight_decay": self.weight_decay[stage],
            # Common parameters
            "bf16": self.bf16,
            "fp16": self.fp16,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "evaluation_strategy": self.evaluation_strategy,
            "save_strategy": self.save_strategy,
            "save_total_limit": self.save_total_limit,
            "seed": self.seed,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_cache": self.use_cache,
            # Additional stage-specific options
            **self.stage_options.get(stage, {})
        }
        
        return config