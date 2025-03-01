"""Tests for the CLI module."""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

from deepseekr1maker.cli.main import main, parse_args, setup_logging

def test_parse_args():
    """Test parse_args function."""
    # Test version argument
    with patch("sys.argv", ["deepseekr1", "--version"]):
        with pytest.raises(SystemExit) as excinfo:
            parse_args()
        assert excinfo.value.code == 0
    
    # Test train command
    args = parse_args(["train", "--config", "config.json", "--output-dir", "output"])
    assert args.command == "train"
    assert args.config == "config.json"
    assert args.output_dir == "output"
    assert args.stages == ["r1zero", "sft", "reasoning_rl", "rejection", "sft2", "distill"]
    
    # Test evaluate command
    args = parse_args(["evaluate", "--model-path", "model", "--dataset", "data.json"])
    assert args.command == "evaluate"
    assert args.model_path == "model"
    assert args.dataset == "data.json"
    assert args.output_file == "evaluation_results.json"
    
    # Test generate command
    args = parse_args(["generate", "--model-path", "model", "--prompt", "Hello"])
    assert args.command == "generate"
    assert args.model_path == "model"
    assert args.prompt == "Hello"
    assert args.output_file == "generations.txt"

def test_setup_logging():
    """Test setup_logging function."""
    # Test with verbose=False
    with patch("logging.basicConfig") as mock_basic_config:
        setup_logging(verbose=False)
        args, kwargs = mock_basic_config.call_args
        assert kwargs["level"] == 20  # INFO level
    
    # Test with verbose=True
    with patch("logging.basicConfig") as mock_basic_config:
        setup_logging(verbose=True)
        args, kwargs = mock_basic_config.call_args
        assert kwargs["level"] == 10  # DEBUG level

def test_main_no_command():
    """Test main function with no command."""
    with patch("deepseekr1maker.cli.main.parse_args") as mock_parse_args:
        mock_args = MagicMock()
        mock_args.command = None
        mock_args.verbose = False
        mock_parse_args.return_value = mock_args
        
        with patch("logging.Logger.error") as mock_error:
            exit_code = main(["--help"])
            assert exit_code == 1
            mock_error.assert_called_once()

def test_main_train_command():
    """Test main function with train command."""
    with patch("deepseekr1maker.cli.main.parse_args") as mock_parse_args:
        mock_args = MagicMock()
        mock_args.command = "train"
        mock_args.config = "config.json"
        mock_args.output_dir = "output"
        mock_args.stages = ["r1zero"]
        mock_args.verbose = False
        mock_parse_args.return_value = mock_args
        
        with patch("logging.Logger.info") as mock_info:
            exit_code = main(["train", "--config", "config.json"])
            assert exit_code == 0
            assert mock_info.call_count >= 2  # At least version and training info

def test_main_evaluate_command():
    """Test main function with evaluate command."""
    with patch("deepseekr1maker.cli.main.parse_args") as mock_parse_args:
        mock_args = MagicMock()
        mock_args.command = "evaluate"
        mock_args.model_path = "model"
        mock_args.dataset = "data.json"
        mock_args.output_file = "results.json"
        mock_args.verbose = False
        mock_parse_args.return_value = mock_args
        
        with patch("logging.Logger.info") as mock_info:
            exit_code = main(["evaluate", "--model-path", "model", "--dataset", "data.json"])
            assert exit_code == 0
            assert mock_info.call_count >= 2  # At least version and evaluation info

def test_main_generate_command():
    """Test main function with generate command."""
    with patch("deepseekr1maker.cli.main.parse_args") as mock_parse_args:
        mock_args = MagicMock()
        mock_args.command = "generate"
        mock_args.model_path = "model"
        mock_args.prompt = "Hello"
        mock_args.prompt_file = None
        mock_args.output_file = "generations.txt"
        mock_args.verbose = False
        mock_parse_args.return_value = mock_args
        
        with patch("logging.Logger.info") as mock_info:
            exit_code = main(["generate", "--model-path", "model", "--prompt", "Hello"])
            assert exit_code == 0
            assert mock_info.call_count >= 2  # At least version and generation info 