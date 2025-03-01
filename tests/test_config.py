"""Tests for the config module."""

import pytest
from deepseekr1maker.config.model_config import TokenizerConfig, GenerationConfig
from deepseekr1maker.config.training_config import TrainingConfig
from deepseekr1maker.config.reward_config import RewardConfig

def test_tokenizer_config():
    """Test TokenizerConfig initialization and defaults."""
    config = TokenizerConfig()
    assert config.use_fast_tokenizer is True
    assert config.padding_side == "right"
    assert config.truncation_side == "right"
    
    # Test custom values
    custom_config = TokenizerConfig(
        tokenizer_name_or_path="gpt2",
        use_fast_tokenizer=False,
        padding_side="left",
        truncation_side="left",
        add_eos_token=False,
        add_bos_token=False
    )
    assert custom_config.tokenizer_name_or_path == "gpt2"
    assert custom_config.use_fast_tokenizer is False
    assert custom_config.padding_side == "left"
    assert custom_config.truncation_side == "left"
    assert custom_config.add_eos_token is False
    assert custom_config.add_bos_token is False

def test_generation_config():
    """Test GenerationConfig initialization and defaults."""
    config = GenerationConfig()
    assert config.temperature == 0.7
    assert config.top_p == 0.9
    
    # Test custom values
    custom_config = GenerationConfig(
        temperature=0.5,
        top_p=0.8,
        top_k=40
    )
    assert custom_config.temperature == 0.5
    assert custom_config.top_p == 0.8
    assert custom_config.top_k == 40

def test_training_config():
    """Test TrainingConfig initialization and defaults."""
    config = TrainingConfig()
    assert config.output_dir == "./deepseek_r1_model"
    assert config.overwrite_output_dir is True
    assert "r1zero" in config.stages
    assert "sft" in config.stages
    
    # Test custom values
    custom_config = TrainingConfig(
        output_dir="./custom_output",
        overwrite_output_dir=False,
        stages=["r1zero", "sft"],
        num_train_epochs={"r1zero": 3, "sft": 4}
    )
    assert custom_config.output_dir == "./custom_output"
    assert custom_config.overwrite_output_dir is False
    assert custom_config.stages == ["r1zero", "sft"]
    assert custom_config.num_train_epochs["r1zero"] == 3
    assert custom_config.num_train_epochs["sft"] == 4

def test_reward_config():
    """Test RewardConfig initialization and defaults."""
    config = RewardConfig()
    assert len(config.reward_funcs) > 0
    
    # Test custom values
    custom_config = RewardConfig(
        reward_funcs=["accuracy", "reasoning_steps"],
        accuracy_weight=0.7,
        reasoning_steps_weight=0.3
    )
    assert custom_config.reward_funcs == ["accuracy", "reasoning_steps"]
    assert custom_config.accuracy_weight == 0.7
    assert custom_config.reasoning_steps_weight == 0.3 