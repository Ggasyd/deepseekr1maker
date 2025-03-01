"""Pytest configuration and fixtures."""

import os
import tempfile
from typing import Dict, Any

import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

@pytest.fixture
def model_config():
    """Create a sample model configuration for testing."""
    from deepseekr1maker.config.model_config import ModelConfig
    
    return ModelConfig(
        model_name_or_path="gpt2",  # Small model for testing
        max_length=128,
        torch_dtype="float32",
        trust_remote_code=True,
        attn_implementation="eager"
    )

@pytest.fixture
def training_config(temp_dir):
    """Create a sample training configuration for testing."""
    from deepseekr1maker.config.training_config import TrainingConfig
    
    return TrainingConfig(
        output_dir=temp_dir,
        overwrite_output_dir=True,
        num_train_epochs={"r1zero": 1, "sft": 1},
        per_device_train_batch_size={"r1zero": 2, "sft": 2},
        per_device_eval_batch_size={"r1zero": 2, "sft": 2},
        gradient_accumulation_steps={"r1zero": 1, "sft": 1},
        learning_rate={"r1zero": 5e-5, "sft": 2e-5},
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        device="cpu"  # Use CPU for testing
    )

@pytest.fixture
def reward_config():
    """Create a sample reward configuration for testing."""
    from deepseekr1maker.config.reward_config import RewardConfig
    
    return RewardConfig(
        reward_funcs=["reasoning_steps", "accuracy"],
        reasoning_steps_weight=0.5,
        accuracy_weight=0.5
    )

@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    from datasets import Dataset
    
    train_data = {
        "input": [
            "What is 2+2?",
            "Explain quantum computing."
        ],
        "output": [
            "2+2=4",
            "Quantum computing uses quantum bits or qubits..."
        ]
    }
    
    eval_data = {
        "input": [
            "What is 3+3?",
            "Explain machine learning."
        ],
        "output": [
            "3+3=6",
            "Machine learning is a subset of AI..."
        ]
    }
    
    return {
        "train": Dataset.from_dict(train_data),
        "validation": Dataset.from_dict(eval_data)
    }

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    return AutoTokenizer.from_pretrained("gpt2")

@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    return AutoModelForCausalLM.from_pretrained("gpt2") 