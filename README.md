# DeepSeek R1 Maker

A Python library that automates the process of creating and training models like DeepSeek R1.

## Features

- Multi-stage training pipeline for large language models
- Comprehensive reward functions for reinforcement learning
- Flexible data loading and preprocessing
- Configurable model architecture and training parameters
- Command-line interface for easy usage

## Installation

### From PyPI

```bash
pip install deepseekr1maker
```

### From Source

```bash
git clone https://github.com/yourusername/deepseekr1maker.git
cd deepseekr1maker
pip install -e .
```

## Quick Start

### Training a Model

```python
from deepseekr1maker.config import ModelConfig, TrainingConfig, RewardConfig
from deepseekr1maker.data import DataLoader
from deepseekr1maker.training import R1ZeroTrainer

# Load configurations
model_config = ModelConfig(model_name_or_path="deepseek-ai/deepseek-coder-1.3b-base")
training_config = TrainingConfig(output_dir="./output")
reward_config = RewardConfig(reward_funcs=["reasoning_steps", "accuracy"])

# Load data
data_loader = DataLoader(r1zero_dataset="path/to/dataset")
dataset = data_loader.prepare_dataset("r1zero")

# Initialize trainer
trainer = R1ZeroTrainer(
    model_config=model_config,
    training_config=training_config,
    reward_config=reward_config,
    dataset=dataset
)

# Train model
trainer.train()

# Save model
trainer.save_model()
```

### Using the CLI

```bash
# Train a model
deepseekr1 train --config config.json --output-dir ./output

# Evaluate a model
deepseekr1 evaluate --model-path ./output/r1zero_final --dataset path/to/eval_dataset

# Generate text
deepseekr1 generate --model-path ./output/r1zero_final --prompt "Write a function to calculate Fibonacci numbers"
```

## Documentation

For detailed documentation, see the [docs](docs/) directory.

## Testing

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## License

This project is licensed under the terms of the license included in the LICENSE file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
