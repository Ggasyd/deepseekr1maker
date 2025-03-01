#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command-line interface for DeepSeek R1 Maker.
"""

import argparse
import logging
import sys
from typing import List, Optional

from deepseekr1maker.__version__ import __version__

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration.
    
    Args:
        verbose: Whether to enable verbose logging
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.
    
    Args:
        args: Command-line arguments (defaults to sys.argv[1:])
        
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="DeepSeek R1 Maker - Automate the process of creating and training models like DeepSeek R1"
    )
    
    parser.add_argument(
        "--version", action="version", version=f"DeepSeek R1 Maker v{__version__}"
    )
    
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--config", type=str, required=True, help="Path to training configuration file"
    )
    train_parser.add_argument(
        "--output-dir", type=str, default="./output", help="Output directory for model and logs"
    )
    train_parser.add_argument(
        "--stages", type=str, nargs="+", 
        default=["r1zero", "sft", "reasoning_rl", "rejection", "sft2", "distill"],
        help="Training stages to execute"
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model")
    eval_parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model to evaluate"
    )
    eval_parser.add_argument(
        "--dataset", type=str, required=True, help="Path to evaluation dataset"
    )
    eval_parser.add_argument(
        "--output-file", type=str, default="evaluation_results.json", 
        help="Output file for evaluation results"
    )
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text with a model")
    gen_parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model to use"
    )
    gen_parser.add_argument(
        "--prompt", type=str, help="Prompt for generation"
    )
    gen_parser.add_argument(
        "--prompt-file", type=str, help="File containing prompts for generation"
    )
    gen_parser.add_argument(
        "--output-file", type=str, default="generations.txt", 
        help="Output file for generations"
    )
    
    return parser.parse_args(args)

def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI.
    
    Args:
        args: Command-line arguments (defaults to sys.argv[1:])
        
    Returns:
        Exit code
    """
    parsed_args = parse_args(args)
    setup_logging(parsed_args.verbose)
    
    logger.info(f"DeepSeek R1 Maker v{__version__}")
    
    if parsed_args.command == "train":
        logger.info(f"Training model with config: {parsed_args.config}")
        # TODO: Implement training logic
        # from ..training import train_model
        # train_model(parsed_args.config, parsed_args.output_dir, parsed_args.stages)
    elif parsed_args.command == "evaluate":
        logger.info(f"Evaluating model: {parsed_args.model_path}")
        # TODO: Implement evaluation logic
        # from ..utils import evaluate_model
        # evaluate_model(parsed_args.model_path, parsed_args.dataset, parsed_args.output_file)
    elif parsed_args.command == "generate":
        logger.info(f"Generating text with model: {parsed_args.model_path}")
        # TODO: Implement generation logic
        # from ..utils import generate_text
        # generate_text(parsed_args.model_path, parsed_args.prompt, parsed_args.prompt_file, parsed_args.output_file)
    else:
        logger.error("No command specified. Use --help for usage information.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 