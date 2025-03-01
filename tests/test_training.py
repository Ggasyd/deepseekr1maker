"""Tests for the training module."""

import os
import pytest
from unittest.mock import patch, MagicMock

from deepseekr1maker.training.r1zero_trainer import R1ZeroTrainer, LoggingCallback

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_r1zero_trainer_init(model_config, training_config, reward_config, sample_dataset, mock_tokenizer, mock_model, device):
    """Test R1ZeroTrainer initialization."""
    # Mock CUDA availability
    with patch("torch.cuda.is_available", return_value=(device == "cuda")):
        # Set device in training config
        training_config.device = device
        
        # Initialize trainer with provided tokenizer and model
        trainer = R1ZeroTrainer(
            model_config=model_config,
            training_config=training_config,
            reward_config=reward_config,
            dataset=sample_dataset,
            tokenizer=mock_tokenizer,
            model=mock_model
        )
        
        # Check that trainer was initialized correctly
        assert trainer.model_config == model_config
        assert trainer.training_config == training_config
        assert trainer.reward_config == reward_config
        assert trainer.dataset == sample_dataset
        assert trainer.tokenizer == mock_tokenizer
        assert trainer.model == mock_model
        assert len(trainer.reward_functions) > 0
        assert trainer.grpo_trainer is None

def test_r1zero_trainer_prepare_training_args(model_config, training_config, reward_config, sample_dataset, mock_tokenizer, mock_model):
    """Test R1ZeroTrainer prepare_training_args method."""
    trainer = R1ZeroTrainer(
        model_config=model_config,
        training_config=training_config,
        reward_config=reward_config,
        dataset=sample_dataset,
        tokenizer=mock_tokenizer,
        model=mock_model
    )
    
    # Check that training args were prepared correctly
    assert trainer.training_args is not None
    assert trainer.training_args.output_dir.endswith("r1zero")
    assert trainer.training_args.learning_rate == training_config.learning_rate.get("r1zero", 5e-5)

@pytest.mark.skip(reason="Requires actual model training which is resource-intensive")
def test_r1zero_trainer_train(model_config, training_config, reward_config, sample_dataset, mock_tokenizer, mock_model):
    """Test R1ZeroTrainer train method."""
    trainer = R1ZeroTrainer(
        model_config=model_config,
        training_config=training_config,
        reward_config=reward_config,
        dataset=sample_dataset,
        tokenizer=mock_tokenizer,
        model=mock_model
    )
    
    # Mock the GRPOTrainer
    mock_grpo_trainer = MagicMock()
    mock_grpo_trainer.train.return_value = {"loss": 0.1}
    
    # Replace the actual trainer with the mock
    with patch("trl.GRPOTrainer", return_value=mock_grpo_trainer):
        result = trainer.train()
        
        # Check that train was called
        mock_grpo_trainer.train.assert_called_once()
        assert result == {"loss": 0.1}

def test_r1zero_trainer_save_model(model_config, training_config, reward_config, sample_dataset, mock_tokenizer, mock_model, temp_dir):
    """Test R1ZeroTrainer save_model method."""
    trainer = R1ZeroTrainer(
        model_config=model_config,
        training_config=training_config,
        reward_config=reward_config,
        dataset=sample_dataset,
        tokenizer=mock_tokenizer,
        model=mock_model
    )
    
    # Mock the GRPOTrainer
    mock_grpo_trainer = MagicMock()
    trainer.grpo_trainer = mock_grpo_trainer
    
    # Test saving with default output directory
    with patch.object(mock_tokenizer, "save_pretrained") as mock_save_tokenizer:
        with patch.object(mock_grpo_trainer, "save_model") as mock_save_model:
            output_dir = trainer.save_model()
            
            # Check that save methods were called
            mock_save_tokenizer.assert_called_once()
            mock_save_model.assert_called_once()
            assert "r1zero_final" in output_dir
    
    # Test saving with custom output directory
    custom_dir = os.path.join(temp_dir, "custom_output")
    with patch.object(mock_tokenizer, "save_pretrained") as mock_save_tokenizer:
        with patch.object(mock_grpo_trainer, "save_model") as mock_save_model:
            output_dir = trainer.save_model(output_dir=custom_dir)
            
            # Check that save methods were called with custom directory
            mock_save_tokenizer.assert_called_once_with(custom_dir)
            mock_save_model.assert_called_once_with(custom_dir)
            assert output_dir == custom_dir

def test_logging_callback():
    """Test LoggingCallback."""
    callback = LoggingCallback()
    
    # Create mock objects
    args = MagicMock()
    args.logging_steps = 10
    
    state = MagicMock()
    state.global_step = 10
    state.log_history = [{"loss": 0.1, "learning_rate": 5e-5}]
    
    control = MagicMock()
    
    # Test callback
    with patch("logging.Logger.info") as mock_info:
        callback.on_step_end(args, state, control)
        mock_info.assert_called_once()
        
    # Test callback with non-logging step
    state.global_step = 11
    with patch("logging.Logger.info") as mock_info:
        callback.on_step_end(args, state, control)
        mock_info.assert_not_called() 