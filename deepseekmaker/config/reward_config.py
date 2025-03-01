from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
#import os
import json

@dataclass
class CosineRewardConfig:
    """Configuration for cosine similarity reward function."""
    min_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Minimum reward value for incorrect answers"}
    )
    max_value_wrong: float = field(
        default=-0.1,
        metadata={"help": "Maximum reward value for incorrect answers"}
    )
    min_value_correct: float = field(
        default=0.8,
        metadata={"help": "Minimum reward value for correct answers"}
    )
    max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward value for correct answers"}
    )
    max_len: int = field(
        default=1000,
        metadata={"help": "Maximum sequence length for cosine scaling"}
    )


@dataclass
class RepetitionPenaltyConfig:
    """Configuration for repetition penalty reward function."""
    n_grams: int = field(
        default=3,
        metadata={"help": "N-gram size for repetition detection"}
    )
    max_penalty: float = field(
        default=-0.1,
        metadata={"help": "Maximum (negative) penalty for repetition"}
    )
    scale_with_length: bool = field(
        default=True,
        metadata={"help": "Scale penalty with sequence length"}
    )
    min_repetitions: int = field(
        default=3,
        metadata={"help": "Minimum number of repetitions to start applying penalty"}
    )


@dataclass
class AccuracyRewardConfig:
    """Configuration for accuracy reward function."""
    exact_match_bonus: float = field(
        default=1.0,
        metadata={"help": "Bonus reward for exact match"}
    )
    partial_match_scale: float = field(
        default=0.5,
        metadata={"help": "Scaling factor for partial matches"}
    )
    use_regex: bool = field(
        default=True,
        metadata={"help": "Use regex for matching"}
    )
    case_sensitive: bool = field(
        default=False,
        metadata={"help": "Case sensitive matching"}
    )
    relaxed_whitespace: bool = field(
        default=True,
        metadata={"help": "Ignore whitespace differences"}
    )


@dataclass
class FormatRewardConfig:
    """Configuration for format reward function."""
    format_templates: Dict[str, str] = field(
        default_factory=lambda: {},
        metadata={"help": "Templates for various format requirements"}
    )
    penalize_wrong_format: float = field(
        default=-0.5,
        metadata={"help": "Penalty for wrong format"}
    )
    bonus_correct_format: float = field(
        default=0.5,
        metadata={"help": "Bonus for correct format"}
    )
    json_validation: bool = field(
        default=True,
        metadata={"help": "Validate JSON syntax if format requires JSON"}
    )


@dataclass
class ReasoningStepsRewardConfig:
    """Configuration for reasoning steps reward function."""
    min_steps_required: int = field(
        default=3,
        metadata={"help": "Minimum number of reasoning steps required"}
    )
    max_steps_rewarded: int = field(
        default=8,
        metadata={"help": "Maximum number of reasoning steps to reward"}
    )
    step_reward: float = field(
        default=0.1,
        metadata={"help": "Reward per identified reasoning step"}
    )
    step_markers: List[str] = field(
        default_factory=lambda: ["First,", "Second,", "Third,", "Finally,", "Step 1:", "Step 2:"],
        metadata={"help": "Markers that indicate reasoning steps"}
    )
    require_conclusion: bool = field(
        default=True,
        metadata={"help": "Require a conclusion section"}
    )


@dataclass
class LanguageConsistencyConfig:
    """Configuration for language consistency reward function."""
    detection_threshold: float = field(
        default=0.95,
        metadata={"help": "Threshold for language detection confidence"}
    )
    target_languages: List[str] = field(
        default_factory=lambda: ["en"],
        metadata={"help": "List of target languages to maintain consistency with"}
    )
    model_name: str = field(
        default="papluca/xlm-roberta-base-language-detection",
        metadata={"help": "Language detection model"}
    )
    penalty_scale: float = field(
        default=-1.0,
        metadata={"help": "Penalty scale for language inconsistency"}
    )





@dataclass
class StageRewardConfig:
    """Configuration for rewards in a specific training stage."""
    reward_funcs: List[str] = field(
        default_factory=list,
        metadata={"help": "List of reward functions to use in this stage"}
    )
    reward_weights: Dict[str, float] = field(
        default_factory=dict,
        metadata={"help": "Weights for each reward function in this stage"}
    )
    kl_coef: float = field(
        default=0.1,
        metadata={"help": "KL divergence coefficient for PPO"}
    )
    gamma: float = field(
        default=0.99,
        metadata={"help": "Discount factor for rewards"}
    )
    lambd: float = field(
        default=0.95,
        metadata={"help": "GAE lambda parameter"}
    )
    clip_range: float = field(
        default=0.2,
        metadata={"help": "PPO clip range"}
    )
    value_clip: float = field(
        default=0.2,
        metadata={"help": "Value function clip range"}
    )
    normalize_rewards: bool = field(
        default=True,
        metadata={"help": "Normalize rewards"}
    )
    reward_scale: float = field(
        default=1.0,
        metadata={"help": "Global scaling factor for rewards"}
    )
    entropy_coef: float = field(
        default=0.0,
        metadata={"help": "Entropy coefficient for training"}
    )


@dataclass
class RewardConfig:
    """Enhanced configuration for the reward functions in DeepSeek R1."""
    # Available reward functions
    available_reward_funcs: List[str] = field(
        default_factory=lambda: [
            "accuracy", "format", "reasoning_steps", "cosine", 
            "repetition_penalty", "language_consistency"
        ],
        metadata={"help": "List of all available reward functions"}
    )
    
    # Default reward functions to use (if not specified per stage)
    reward_funcs: List[str] = field(
        default_factory=lambda: ["accuracy", "format", "reasoning_steps", "cosine", "repetition_penalty", "language_consistency"],
        metadata={"help": "List of reward functions to use"}
    )
    
    # Default weights for each reward function (if not specified per stage)
    reward_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "accuracy": 1.0,
            "format": 0.8,
            "reasoning_steps": 0.6,
            "cosine": 0.4,
            "repetition_penalty": 0.2,
            "language_consistency": 0.8
        },
        metadata={"help": "Weights for each reward function"}
    )
    
    # Stage-specific reward configurations
    stage_configs: Dict[str, StageRewardConfig] = field(
        default_factory=lambda: {
            "reasoning_rl": StageRewardConfig(
                reward_funcs=["accuracy", "reasoning_steps", "cosine", "language_consistency"],
                reward_weights={
                    "accuracy": 1.0,
                    "reasoning_steps": 0.8,
                    "cosine": 0.6,
                    "language_consistency": 0.9
                },
                kl_coef=0.05
            ),
            "rejection": StageRewardConfig(
                reward_funcs=["accuracy", "format", "repetition_penalty"],
                reward_weights={
                    "accuracy": 1.0,
                    "format": 0.7,
                    "repetition_penalty": 0.4
                },
                kl_coef=0.1
            )
        },
        metadata={"help": "Stage-specific reward configurations"}
    )
    
    # Specific reward function configurations
    cosine: CosineRewardConfig = field(
        default_factory=CosineRewardConfig,
        metadata={"help": "Configuration for cosine similarity reward"}
    )
    
    repetition_penalty: RepetitionPenaltyConfig = field(
        default_factory=RepetitionPenaltyConfig,
        metadata={"help": "Configuration for repetition penalty"}
    )
    
    accuracy: AccuracyRewardConfig = field(
        default_factory=AccuracyRewardConfig,
        metadata={"help": "Configuration for accuracy reward"}
    )
    
    format: FormatRewardConfig = field(
        default_factory=FormatRewardConfig,
        metadata={"help": "Configuration for format reward"}
    )
    
    reasoning_steps: ReasoningStepsRewardConfig = field(
        default_factory=ReasoningStepsRewardConfig,
        metadata={"help": "Configuration for reasoning steps reward"}
    )
    
    language_consistency: LanguageConsistencyConfig = field(
        default_factory=LanguageConsistencyConfig,
        metadata={"help": "Configuration for language consistency"}
    )
    
    # Global reward settings
    reward_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for reward model inference"}
    )
    
    reward_device: str = field(
        default="cuda",
        metadata={"help": "Device for reward model computation"}
    )
    
    combine_method: str = field(
        default="weighted_sum",
        metadata={"help": "Method to combine multiple rewards: 'weighted_sum', 'min', 'max', 'product'"}
    )
    
    cache_rewards: bool = field(
        default=True,
        metadata={"help": "Whether to cache rewards for identical responses"}
    )
    
    reward_debug_mode: bool = field(
        default=False,
        metadata={"help": "Enable logging of individual reward components for debugging"}
    )
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Check if all reward_funcs are in available_reward_funcs
        for func in self.reward_funcs:
            if func not in self.available_reward_funcs:
                raise ValueError(f"Reward function '{func}' is not available.")
        
        # Check if all reward_funcs have weights
        missing_weights = set(self.reward_funcs) - set(self.reward_weights.keys())
        if missing_weights:
            raise ValueError(f"Missing weights for reward functions: {missing_weights}")
        
        # Check if all stages have valid configs
        for stage, config in self.stage_configs.items():
            for func in config.reward_funcs:
                if func not in self.available_reward_funcs:
                    raise ValueError(f"Reward function '{func}' used in stage '{stage}' is not available.")
                
                if func not in config.reward_weights:
                    # Use default weight if not specified in stage config
                    if func in self.reward_weights:
                        config.reward_weights[func] = self.reward_weights[func]
                    else:
                        raise ValueError(f"Missing weight for reward function '{func}' in stage '{stage}'.")
    
    def get_stage_config(self, stage: str) -> StageRewardConfig:
        """Get reward configuration for a specific stage."""
        if stage in self.stage_configs:
            return self.stage_configs[stage]
        
        # Create a default stage config if not defined
        default_config = StageRewardConfig(
            reward_funcs=self.reward_funcs,
            reward_weights={func: self.reward_weights[func] for func in self.reward_funcs}
        )
        
        return default_config
    

    
    def save_config(self, output_path: str):
        """Save the reward configuration to a JSON file."""
        # Convert to dictionary (excluding methods)
        config_dict = {k: v for k, v in self.__dict__.items() if not callable(v)}
        
        # Convert nested dataclasses to dictionaries
        for key, value in config_dict.items():
            if hasattr(value, "__dataclass_fields__"):
                config_dict[key] = {k: v for k, v in value.__dict__.items()}
        
        # Convert stage configs
        if "stage_configs" in config_dict:
            stage_dict = {}
            for stage_name, stage_config in config_dict["stage_configs"].items():
                stage_dict[stage_name] = {k: v for k, v in stage_config.__dict__.items()}
            config_dict["stage_configs"] = stage_dict
        
        # Save to file
        with open(output_path, "w") as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_config(cls, config_path: str) -> "RewardConfig":
        """Load reward configuration from a JSON file."""
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Create basic config
        reward_config = cls()
        
        # Set top-level primitives
        for key, value in config_dict.items():
            if key not in ["cosine", "repetition_penalty", "accuracy", "format", 
                          "reasoning_steps", "language_consistency", "custom", "stage_configs"]:
                if hasattr(reward_config, key):
                    setattr(reward_config, key, value)
        
        # Set nested configurations
        for key in ["cosine", "repetition_penalty", "accuracy", "format", 
                   "reasoning_steps", "language_consistency", "custom"]:
            if key in config_dict:
                sub_config = getattr(reward_config, key)
                for sub_key, sub_value in config_dict[key].items():
                    if hasattr(sub_config, sub_key):
                        setattr(sub_config, sub_key, sub_value)
        
        # Set stage configurations
        if "stage_configs" in config_dict:
            reward_config.stage_configs = {}
            for stage_name, stage_dict in config_dict["stage_configs"].items():
                stage_config = StageRewardConfig()
                for key, value in stage_dict.items():
                    if hasattr(stage_config, key):
                        setattr(stage_config, key, value)
                reward_config.stage_configs[stage_name] = stage_config
        
        return reward_config