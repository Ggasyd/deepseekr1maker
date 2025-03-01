# deepseekr1maker/training/r1zero_trainer.py
import os
import torch
import logging
from typing import Dict, List, Any, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from trl import GRPOTrainer, GRPOConfig
from deepseekr1maker.rewards import get_reward_functions

logger = logging.getLogger(__name__)

class LoggingCallback(TrainerCallback):
    """
    Simple callback to log training information at specific steps.
    """
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % args.logging_steps == 0:
            logger.info(f"Step {state.global_step}: Loss = {state.log_history[-1].get('loss', None)}, "
                      f"Learning rate = {state.log_history[-1].get('learning_rate', None)}")

class R1ZeroTrainer:
    """
    Trainer for the R1 Zero step with the GRPO algorithm.
    """
    
    def __init__(
        self,
        model_config,
        training_config,
        reward_config,
        dataset,
        tokenizer=None,
        model=None,
        callbacks=None
    ):
        """
        Initialize the R1 Zero trainer.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
            reward_config: Reward functions configuration
            dataset: Prepared dataset for training
            tokenizer: Pre-loaded tokenizer (optional)
            model: Pre-loaded model (optional)
            callbacks: List of callbacks (optional)
        """
        self.model_config = model_config
        self.training_config = training_config
        self.reward_config = reward_config
        self.dataset = dataset
        self.callbacks = callbacks or []
        
        # Load tokenizer if not provided
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_config.model_name_or_path,
                trust_remote_code=model_config.trust_remote_code,
                padding_side="right"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer
        
        # Load model if not provided
        if model is None:
            # Determine dtype
            if model_config.torch_dtype == "float16":
                dtype = torch.float16
            elif model_config.torch_dtype == "bfloat16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
            
            # Load the model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_config.model_name_or_path,
                torch_dtype=dtype,
                trust_remote_code=model_config.trust_remote_code,
                attn_implementation=model_config.attn_implementation
            )
        else:
            self.model = model
        
        # Move model to appropriate device
        self.device = training_config.device
        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
        
        # Prepare training arguments
        self.prepare_training_args()
        
        # Initialize reward functions
        self.reward_functions = get_reward_functions(reward_config)
        
        # Initialize GRPO trainer
        self.grpo_trainer = None
    
    def prepare_training_args(self):
        """Prepare training arguments from the training configuration."""
        # Output directory
        output_dir = os.path.join(
            self.training_config.output_dir, 
            "r1zero"
        )
        
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=self.training_config.overwrite_output_dir,
            num_train_epochs=self.training_config.num_train_epochs.get("r1zero", 1),
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate.get("r1zero", 5e-5),
            warmup_ratio=self.training_config.warmup_ratio,
            weight_decay=self.training_config.weight_decay,
            logging_steps=self.training_config.logging_steps,
            evaluation_strategy=self.training_config.evaluation_strategy,
            eval_steps=self.training_config.eval_steps,
            save_strategy=self.training_config.save_strategy,
            save_steps=self.training_config.save_steps,
            save_total_limit=self.training_config.save_total_limit,
            seed=self.training_config.seed,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
        )
    
    def train(self):
        """
        Run the R1 Zero training with GRPO.
        
        Returns:
            Training results
        """
        logger.info("Initializing R1 Zero training with GRPO...")
        
        # Set up GRPO configuration
        grpo_config = GRPOConfig(
            # GRPO-specific parameters
            target_kl=0.1,  # Target KL divergence (adjust as needed)
            kl_penalty_factor=0.5,  # KL penalty factor (adjust as needed)
            max_sequence_length=self.model_config.max_length,
            use_factual_completions=True,
            reward_functions=self.reward_functions,
        )
        
        # Initialize GRPO trainer
        self.grpo_trainer = GRPOTrainer(
            model=self.model,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset.get("validation"),
            tokenizer=self.tokenizer,
            args=self.training_args,
            callbacks=self.callbacks,
            grpo_config=grpo_config,
        )
        
        logger.info("Starting R1 Zero training...")
        train_result = self.grpo_trainer.train()
        
        logger.info(f"R1 Zero training completed. Results: {train_result}")
        return train_result
    
    def save_model(self, output_dir=None):
        """
        Save the trained model and tokenizer.
        
        Args:
            output_dir: Output directory (optional, uses configuration directory by default)
        """
        if output_dir is None:
            output_dir = os.path.join(
                self.training_config.output_dir, 
                "r1zero_final"
            )
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        logger.info(f"Saving R1 Zero model to {output_dir}...")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save model via trainer
        if self.grpo_trainer:
            self.grpo_trainer.save_model(output_dir)
        else:
            # If trainer hasn't been initialized, save model directly
            self.model.save_pretrained(output_dir)
        
        logger.info(f"R1 Zero model successfully saved to {output_dir}")
        
        return output_dir