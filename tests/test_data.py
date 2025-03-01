"""Tests for the data module."""

import os
import pytest
from unittest.mock import patch, MagicMock
from datasets import Dataset, DatasetDict

from deepseekr1maker.data.data_loader import DataLoader

def test_data_loader_init():
    """Test DataLoader initialization."""
    # Test with default parameters
    loader = DataLoader()
    assert loader.validation_split == 0.1
    assert loader.test_split == 0.0
    assert loader.use_predefined_splits is True
    
    # Test with custom parameters
    custom_loader = DataLoader(
        validation_split=0.2,
        test_split=0.1,
        use_predefined_splits=False
    )
    assert custom_loader.validation_split == 0.2
    assert custom_loader.test_split == 0.1
    assert custom_loader.use_predefined_splits is False

def test_data_loader_with_datasets():
    """Test DataLoader with provided datasets."""
    # Create sample datasets
    train_data = Dataset.from_dict({
        "input": ["What is 2+2?", "Explain quantum computing."],
        "output": ["2+2=4", "Quantum computing uses quantum bits or qubits..."]
    })
    
    eval_data = Dataset.from_dict({
        "input": ["What is 3+3?", "Explain machine learning."],
        "output": ["3+3=6", "Machine learning is a subset of AI..."]
    })
    
    # Create dataset dict
    dataset_dict = DatasetDict({
        "train": train_data,
        "validation": eval_data
    })
    
    # Test with dataset dict
    loader = DataLoader(datasets={"r1zero": dataset_dict})
    assert "r1zero" in loader.datasets
    
    # Test with direct dataset assignment
    loader = DataLoader(r1zero_dataset=dataset_dict)
    assert loader.datasets.get("r1zero") == dataset_dict

@pytest.mark.parametrize("stage", ["r1zero", "sft", "reasoning_rl", "rejection", "sft2", "distill"])
def test_prepare_dataset_with_mock(stage):
    """Test prepare_dataset method with mock datasets."""
    # Create mock dataset
    mock_dataset = MagicMock(spec=DatasetDict)
    mock_dataset.__getitem__.return_value = MagicMock(spec=Dataset)
    
    # Create data loader with mock dataset
    loader = DataLoader(**{f"{stage}_dataset": mock_dataset})
    
    # Mock column mapping
    column_mapping = {
        "input": "instruction",
        "output": "response"
    }
    loader.column_mappings = {stage: column_mapping}
    
    # Mock preprocessing function
    def mock_preprocess(examples):
        return examples
    
    with patch.object(loader, "_preprocess_dataset", side_effect=mock_preprocess):
        # Prepare dataset
        prepared_dataset = loader.prepare_dataset(stage)
        
        # Check that dataset was prepared
        assert prepared_dataset is not None

def test_data_loader_with_system_prompts():
    """Test DataLoader with system prompts."""
    # Create sample dataset
    dataset = Dataset.from_dict({
        "input": ["What is 2+2?", "Explain quantum computing."],
        "output": ["2+2=4", "Quantum computing uses quantum bits or qubits..."]
    })
    
    # Create system prompts
    system_prompts = {
        "r1zero": "You are a helpful assistant.",
        "sft": "You are a knowledgeable AI."
    }
    
    # Create data loader
    loader = DataLoader(
        r1zero_dataset=dataset,
        system_prompts=system_prompts
    )
    
    # Check that system prompts were set
    assert loader.system_prompts == system_prompts
    assert loader.system_prompts.get("r1zero") == "You are a helpful assistant."
    assert loader.system_prompts.get("sft") == "You are a knowledgeable AI." 