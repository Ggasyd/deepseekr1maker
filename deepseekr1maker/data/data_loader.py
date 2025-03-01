# deepseekr1maker/data/data_loader.py
import os
import pandas as pd
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datasets import load_dataset, Dataset, DatasetDict
from functools import partial
import random
import re
import copy

logger = logging.getLogger(__name__)

class DataLoader:
    """Class for loading and preparing training data for different stages of DeepSeek R1 training."""
    
    def __init__(
        self,
        # Base dataset options
        datasets: Optional[Dict[str, Union[str, Dataset, DatasetDict]]] = None,
        
        # Stage-specific dataset paths (if not specified in datasets dict)
        r1zero_dataset: Optional[Union[str, Dataset, DatasetDict]] = None,
        sft_dataset: Optional[Union[str, Dataset, DatasetDict]] = None,
        reasoning_rl_dataset: Optional[Union[str, Dataset, DatasetDict]] = None,
        rejection_dataset: Optional[Union[str, Dataset, DatasetDict]] = None,
        sft2_dataset: Optional[Union[str, Dataset, DatasetDict]] = None,
        distill_dataset: Optional[Union[str, Dataset, DatasetDict]] = None,
        
        # Column mappings for each stage
        column_mappings: Optional[Dict[str, Dict[str, str]]] = None,
        
        # System prompts for different stages
        system_prompts: Optional[Dict[str, str]] = None,
        
        # Cold start options
        generate_cold_start: bool = False,
        cold_start_model: Optional[str] = None,
        cold_start_config: Optional[Dict[str, Any]] = None,
        
        # Preprocessing options
        preprocessing_config: Optional[Dict[str, Any]] = None,
        
        # Data split options
        validation_split: float = 0.1,
        test_split: float = 0.0,
        use_predefined_splits: bool = True,
        
        # Limit options
        max_samples: Optional[Dict[str, int]] = None,
        
        # Cache options
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        
        # Other options
        streaming: bool = False,
        seed: int = 42,
        verbose: bool = False
    ):
        """
        Initialize the data loader.
        
        Args:
            datasets: Dictionary mapping stage names to dataset paths/objects
            r1zero_dataset: Path or dataset object for R1 Zero training
            sft_dataset: Path or dataset object for SFT training
            reasoning_rl_dataset: Path or dataset object for reasoning RL training
            rejection_dataset: Path or dataset object for rejection sampling
            sft2_dataset: Path or dataset object for second SFT round
            distill_dataset: Path or dataset object for distillation
            column_mappings: Dictionary of column mappings for each dataset stage
            system_prompts: Dictionary of system prompts for each stage
            generate_cold_start: Generate automatic Cold Start data if True
            cold_start_model: Model to use for cold start data generation
            cold_start_config: Configuration for cold start data generation
            preprocessing_config: Additional preprocessing configuration
            validation_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            use_predefined_splits: Use predefined splits in datasets if available
            max_samples: Maximum number of samples to load for each stage
            cache_dir: Directory for caching datasets
            use_cache: Whether to use cache for datasets
            streaming: Whether to use streaming mode for large datasets
            seed: Random seed for reproducibility
            verbose: Enable verbose logging
        """
        # Set up logging
        self.verbose = verbose
        if verbose:
            logging.basicConfig(level=logging.INFO)
        
        # Initialize stage datasets
        self.datasets = datasets or {}
        
        # Add individual stage datasets if provided
        if r1zero_dataset:
            self.datasets['r1zero'] = r1zero_dataset
        if sft_dataset:
            self.datasets['sft'] = sft_dataset
        if reasoning_rl_dataset:
            self.datasets['reasoning_rl'] = reasoning_rl_dataset
        if rejection_dataset:
            self.datasets['rejection'] = rejection_dataset
        if sft2_dataset:
            self.datasets['sft2'] = sft2_dataset
        if distill_dataset:
            self.datasets['distill'] = distill_dataset
        
        # Default column mappings
        default_column_mappings = {
            'r1zero': {
                'problem': 'problem',
                'solution': 'solution'
            },
            'sft': {
                'instruction': 'instruction',
                'input': 'input',
                'output': 'output'
            },
            'reasoning_rl': {
                'instruction': 'instruction',
                'input': 'input',
                'output': 'output'
            },
            'rejection': {
                'instruction': 'instruction', 
                'input': 'input',
                'good_output': 'good_output',
                'bad_output': 'bad_output'
            },
            'sft2': {
                'instruction': 'instruction',
                'input': 'input',
                'output': 'output'
            },
            'distill': {
                'instruction': 'instruction',
                'input': 'input',
                'output': 'output',
                'teacher_output': 'teacher_output'
            }
        }
        
        # Merge with user-provided column mappings
        self.column_mappings = default_column_mappings
        if column_mappings:
            for stage, mapping in column_mappings.items():
                if stage in self.column_mappings:
                    self.column_mappings[stage].update(mapping)
                else:
                    self.column_mappings[stage] = mapping
        
        # Default system prompts
        default_system_prompts = {
            'r1zero': (
                "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
                "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
                "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                "<think> reasoning process here </think><answer> answer here </answer>"
            ),
            'sft': (
                "You are an AI assistant specialized in step-by-step reasoning. When presented with a problem, "
                "first analyze it carefully, break it down into manageable steps, and solve each step "
                "methodically. Mark your reasoning process with clearly numbered steps and provide a clear "
                "summary of your final answer."
            ),
            'reasoning_rl': (
                "You are DeepSeek R1, an AI assistant specialized in step-by-step reasoning. When presented with a problem, "
                "break it down into manageable steps, solving each methodically. Always show your work clearly."
            ),
            'rejection': (
                "You are DeepSeek R1, an AI assistant that provides helpful, accurate, and safe responses."
            ),
            'sft2': (
                "You are DeepSeek R1, an AI assistant specialized in precise and comprehensive reasoning. "
                "Provide step-by-step solutions that are clear, accurate, and well-structured."
            ),
            'distill': (
                "You are DeepSeek R1, an AI assistant that provides helpful, detailed, and accurate responses "
                "to user queries."
            )
        }
        
        # Merge with user-provided system prompts
        self.system_prompts = default_system_prompts
        if system_prompts:
            self.system_prompts.update(system_prompts)
        
        # Cold start settings
        self.generate_cold_start = generate_cold_start
        self.cold_start_model = cold_start_model
        self.cold_start_config = cold_start_config or {}
        
        # General settings
        self.preprocessing_config = preprocessing_config or {}
        self.validation_split = validation_split
        self.test_split = test_split
        self.use_predefined_splits = use_predefined_splits
        self.max_samples = max_samples or {}
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.streaming = streaming
        self.seed = seed
        
        # Attributes to initialize later
        self.prepared_datasets = {}
    
    def _load_dataset_from_source(self, source: Union[str, Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
        """
        Load a dataset from various sources (local file, HF dataset ID, or dataset object).
        
        Args:
            source: Path to dataset, Hugging Face dataset ID, or dataset object
            
        Returns:
            Loaded dataset as Dataset or DatasetDict
        """
        # If already a dataset object, return it
        if isinstance(source, (Dataset, DatasetDict)):
            return source
        
        # If it's a string, it could be a local file path or a HF dataset ID
        if isinstance(source, str):
            # Check if it's a local file
            if os.path.exists(source):
                return self._load_local_file(source)
            
            # Otherwise, try to load as Hugging Face dataset
            try:
                load_kwargs = {}
                if self.cache_dir:
                    load_kwargs['cache_dir'] = self.cache_dir
                if not self.use_cache:
                    load_kwargs['download_mode'] = 'force_redownload'
                if self.streaming:
                    load_kwargs['streaming'] = True
                
                dataset = load_dataset(source, **load_kwargs)
                if self.verbose:
                    logger.info(f"Loaded dataset from Hugging Face: {source}")
                return dataset
            except Exception as e:
                raise ValueError(f"Failed to load dataset from '{source}'. Error: {str(e)}")
        
        raise TypeError(f"Unsupported dataset source type: {type(source)}")
    
    def _load_local_file(self, file_path: str) -> Union[Dataset, DatasetDict]:
        """
        Load a dataset from a local file based on its extension.
        
        Args:
            file_path: Path to the local file
            
        Returns:
            Loaded dataset
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine the format based on file extension
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == '.csv':
                df = pd.read_csv(file_path)
                if self.verbose:
                    logger.info(f"Loaded CSV file with {len(df)} rows: {file_path}")
                return Dataset.from_pandas(df)
            
            elif ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Determine if it's a list of items or a dictionary with splits
                if isinstance(data, list):
                    if self.verbose:
                        logger.info(f"Loaded JSON file with {len(data)} items: {file_path}")
                    return Dataset.from_list(data)
                elif isinstance(data, dict) and any(k in data for k in ['train', 'validation', 'test']):
                    splits = {}
                    for split_name, split_data in data.items():
                        if isinstance(split_data, list):
                            splits[split_name] = Dataset.from_list(split_data)
                    if self.verbose:
                        split_info = ', '.join([f"{k}: {len(v)}" for k, v in splits.items()])
                        logger.info(f"Loaded JSON file with splits {split_info}: {file_path}")
                    return DatasetDict(splits)
                else:
                    if self.verbose:
                        logger.info(f"Loaded JSON file as dictionary: {file_path}")
                    return Dataset.from_dict(data)
            
            elif ext == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = [json.loads(line) for line in f]
                if self.verbose:
                    logger.info(f"Loaded JSONL file with {len(data)} items: {file_path}")
                return Dataset.from_list(data)
            
            elif ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                if self.verbose:
                    logger.info(f"Loaded TXT file with {len(lines)} lines: {file_path}")
                return Dataset.from_dict({"text": lines})
            
            else:
                # Try to load with Hugging Face's load_dataset
                dataset = load_dataset(file_path, cache_dir=self.cache_dir)
                if self.verbose:
                    logger.info(f"Loaded dataset using load_dataset: {file_path}")
                return dataset
                
        except Exception as e:
            raise ValueError(f"Failed to load dataset from '{file_path}'. Error: {str(e)}")
    
    def _ensure_splits(self, dataset: Union[Dataset, DatasetDict], stage: str) -> DatasetDict:
        """
        Ensure the dataset has train, validation (and optionally test) splits.
        
        Args:
            dataset: Dataset to split
            stage: Training stage name
            
        Returns:
            DatasetDict with appropriate splits
        """
        # If already a DatasetDict with the required splits, use it
        if isinstance(dataset, DatasetDict):
            if self.use_predefined_splits and 'train' in dataset and 'validation' in dataset:
                if self.test_split > 0 and 'test' not in dataset:
                    # Split validation to create test set
                    test_size = self.test_split / (self.test_split + self.validation_split)
                    test_valid_split = dataset['validation'].train_test_split(
                        test_size=test_size, shuffle=True, seed=self.seed
                    )
                    dataset = DatasetDict({
                        'train': dataset['train'],
                        'validation': test_valid_split['train'],
                        'test': test_valid_split['test']
                    })
                return dataset
        
        # Convert Dataset to DatasetDict with splits
        if isinstance(dataset, Dataset):
            if self.test_split > 0:
                # Three-way split
                train_valid_test = dataset.train_test_split(
                    test_size=self.validation_split + self.test_split,
                    shuffle=True, 
                    seed=self.seed
                )
                
                # Further split the test portion into validation and test
                test_size = self.test_split / (self.validation_split + self.test_split)
                valid_test_split = train_valid_test['test'].train_test_split(
                    test_size=test_size,
                    shuffle=True,
                    seed=self.seed
                )
                
                return DatasetDict({
                    'train': train_valid_test['train'],
                    'validation': valid_test_split['train'],
                    'test': valid_test_split['test']
                })
            else:
                # Two-way split
                train_valid = dataset.train_test_split(
                    test_size=self.validation_split,
                    shuffle=True,
                    seed=self.seed
                )
                
                return DatasetDict({
                    'train': train_valid['train'],
                    'validation': train_valid['test']
                })
        
        # If not a Dataset or DatasetDict, raise an error
        raise ValueError(f"Dataset for stage '{stage}' must be a Dataset or DatasetDict.")
    
    def _limit_samples(self, dataset: DatasetDict, stage: str) -> DatasetDict:
        """
        Limit the number of samples in each split of the dataset.
        
        Args:
            dataset: Dataset to limit
            stage: Training stage name
            
        Returns:
            Dataset with limited samples
        """
        if stage not in self.max_samples:
            return dataset
        
        max_count = self.max_samples[stage]
        if max_count is None or max_count <= 0:
            return dataset
        
        result = DatasetDict()
        
        for split_name, split_dataset in dataset.items():
            if split_name == 'train':
                # Take the specified maximum number for training
                count = min(len(split_dataset), max_count)
                result[split_name] = split_dataset.select(range(count))
                if self.verbose:
                    logger.info(f"Limited {stage} {split_name} split to {count} samples")
            else:
                # For validation and test, keep a proportional amount
                split_ratio = 0
                if split_name == 'validation':
                    split_ratio = self.validation_split
                elif split_name == 'test':
                    split_ratio = self.test_split
                
                if split_ratio > 0:
                    count = min(len(split_dataset), max(1, int(max_count * split_ratio)))
                    result[split_name] = split_dataset.select(range(count))
                    if self.verbose:
                        logger.info(f"Limited {stage} {split_name} split to {count} samples")
                else:
                    result[split_name] = split_dataset
        
        return result
    
    def _check_required_columns(self, dataset: DatasetDict, stage: str) -> None:
        """
        Check if the dataset has all the required columns for the stage.
        
        Args:
            dataset: Dataset to check
            stage: Training stage name
            
        Raises:
            ValueError: If required columns are missing
        """
        if stage not in self.column_mappings:
            return
        
        required_columns = list(self.column_mappings[stage].values())
        for split_name, split_dataset in dataset.items():
            missing = [col for col in required_columns if col not in split_dataset.column_names]
            if missing:
                error_msg = (f"Missing columns in dataset for stage '{stage}', split '{split_name}': {missing}. "
                             f"Available columns: {split_dataset.column_names}")
                if stage in self.column_mappings:
                    error_msg += f"\nConsider updating column_mappings for stage '{stage}'."
                raise ValueError(error_msg)
    
    def _prepare_r1zero_data(self, dataset: DatasetDict) -> DatasetDict:
        """
        Prepare data for R1 Zero training.
        
        Args:
            dataset: Dataset to prepare
            
        Returns:
            Prepared dataset
        """
        columns = self.column_mappings['r1zero']
        problem_col = columns['problem']
        solution_col = columns['solution']
        
        def format_r1zero(example):
            return {
                "prompt": [
                    {"role": "system", "content": self.system_prompts['r1zero']},
                    {"role": "user", "content": example[problem_col]},
                ],
                "solution": example[solution_col]
            }
        
        # Apply the transformation
        prepared_dataset = dataset.map(format_r1zero)
        
        # Validate the dataset
        sample = prepared_dataset['train'][0]
        messages = sample['prompt']
        
        if (len(messages) < 2 or
            messages[0]['role'] != 'system' or
            messages[1]['role'] != 'user'):
            raise ValueError("Incorrect prompt format after R1 Zero transformation.")
        
        if self.verbose:
            logger.info(f"Prepared R1 Zero dataset: {len(prepared_dataset['train'])} training samples")
        
        return prepared_dataset
    
    def _prepare_sft_data(self, dataset: DatasetDict, stage: str = 'sft') -> DatasetDict:
        """
        Prepare data for SFT, SFT2 or distillation training.
        
        Args:
            dataset: Dataset to prepare
            stage: Training stage ('sft', 'sft2', or 'distill')
            
        Returns:
            Prepared dataset
        """
        columns = self.column_mappings[stage]
        instruction_col = columns.get('instruction', 'instruction')
        input_col = columns.get('input', 'input')
        output_col = columns.get('output', 'output')
        teacher_output_col = columns.get('teacher_output', None) if stage == 'distill' else None
        
        def format_sft(example):
            instruction = example.get(instruction_col, '')
            user_input = example.get(input_col, '')
            
            # Combine instruction and input if both exist
            if instruction and user_input:
                user_content = f"{instruction}\n\n{user_input}"
            elif instruction:
                user_content = instruction
            else:
                user_content = user_input
            
            result = {
                "prompt": [
                    {"role": "system", "content": self.system_prompts[stage]},
                    {"role": "user", "content": user_content},
                ],
                "completion": [
                    {"role": "assistant", "content": example[output_col]}
                ]
            }
            
            # Add teacher output for distillation if available
            if stage == 'distill' and teacher_output_col and teacher_output_col in example:
                result["teacher_completion"] = [
                    {"role": "assistant", "content": example[teacher_output_col]}
                ]
            
            return result
        
        # Apply the transformation
        prepared_dataset = dataset.map(format_sft)
        
        # Validate the dataset
        sample = prepared_dataset['train'][0]
        if stage == 'distill' and teacher_output_col and teacher_output_col in dataset['train'][0]:
            if 'teacher_completion' not in sample or not sample['teacher_completion']:
                raise ValueError(f"Missing teacher_completion after {stage} transformation.")
        
        if self.verbose:
            logger.info(f"Prepared {stage} dataset: {len(prepared_dataset['train'])} training samples")
        
        return prepared_dataset
    
    def _prepare_reasoning_rl_data(self, dataset: DatasetDict) -> DatasetDict:
        """
        Prepare data for Reasoning RL training.
        
        Args:
            dataset: Dataset to prepare
            
        Returns:
            Prepared dataset
        """
        # For RL, we use a similar format to SFT but may need extra fields
        prepared_dataset = self._prepare_sft_data(dataset, stage='reasoning_rl')
        
        def add_rl_fields(example):
            # Add any additional fields needed for RL (empty placeholders for now)
            example['reward'] = 0.0
            return example
        
        prepared_dataset = prepared_dataset.map(add_rl_fields)
        
        if self.verbose:
            logger.info(f"Prepared Reasoning RL dataset: {len(prepared_dataset['train'])} training samples")
        
        return prepared_dataset
    
    def _prepare_rejection_data(self, dataset: DatasetDict) -> DatasetDict:
        """
        Prepare data for rejection sampling training.
        
        Args:
            dataset: Dataset to prepare
            
        Returns:
            Prepared dataset
        """
        columns = self.column_mappings['rejection']
        instruction_col = columns.get('instruction', 'instruction')
        input_col = columns.get('input', 'input')
        good_output_col = columns.get('good_output', 'good_output')
        bad_output_col = columns.get('bad_output', 'bad_output')
        
        def format_rejection(example):
            instruction = example.get(instruction_col, '')
            user_input = example.get(input_col, '')
            
            # Combine instruction and input if both exist
            if instruction and user_input:
                user_content = f"{instruction}\n\n{user_input}"
            elif instruction:
                user_content = instruction
            else:
                user_content = user_input
            
            return {
                "prompt": [
                    {"role": "system", "content": self.system_prompts['rejection']},
                    {"role": "user", "content": user_content},
                ],
                "good_completion": [
                    {"role": "assistant", "content": example[good_output_col]}
                ],
                "bad_completion": [
                    {"role": "assistant", "content": example[bad_output_col]}
                ]
            }
        
        # Apply the transformation
        prepared_dataset = dataset.map(format_rejection)
        
        if self.verbose:
            logger.info(f"Prepared Rejection Sampling dataset: {len(prepared_dataset['train'])} training samples")
        
        return prepared_dataset
    
    def _generate_cold_start_data(self) -> Optional[DatasetDict]:
        """
        Generate cold start data using a base model.
        
        Returns:
            Generated dataset or None if generation fails
        """
        if not self.generate_cold_start or not self.cold_start_model:
            return None
        
        if self.verbose:
            logger.info("Cold start data generation not implemented yet. Please provide a dataset.")
        
        # This would require integrating with a model to generate examples
        # For now, this is a placeholder
        raise NotImplementedError(
            "Automatic cold start data generation is not implemented yet. "
            "Please provide an explicit dataset for each stage."
        )
    
    def _apply_preprocessing(self, dataset: DatasetDict, stage: str) -> DatasetDict:
        """
        Apply preprocessing transformations to the dataset.
        
        Args:
            dataset: Dataset to preprocess
            stage: Training stage name
            
        Returns:
            Preprocessed dataset
        """
        if not self.preprocessing_config:
            return dataset
        
        # Apply stage-specific preprocessing if defined
        stage_preprocessing = self.preprocessing_config.get(stage, {})
        if not stage_preprocessing:
            return dataset
        
        # Apply preprocessing functions
        processed_dataset = copy.deepcopy(dataset)
        
        # Apply text cleaning if specified
        if stage_preprocessing.get('clean_text', False):
            def clean_text(text):
                if not text:
                    return text
                # Basic text cleaning (customize as needed)
                text = re.sub(r'\s+', ' ', text)
                text = text.strip()
                return text
            
            def apply_cleaning(example):
                if stage == 'r1zero':
                    if 'prompt' in example:
                        for i, msg in enumerate(example['prompt']):
                            if 'content' in msg:
                                example['prompt'][i]['content'] = clean_text(msg['content'])
                    if 'solution' in example:
                        example['solution'] = clean_text(example['solution'])
                else:
                    if 'prompt' in example:
                        for i, msg in enumerate(example['prompt']):
                            if 'content' in msg:
                                example['prompt'][i]['content'] = clean_text(msg['content'])
                    if 'completion' in example:
                        for i, msg in enumerate(example['completion']):
                            if 'content' in msg:
                                example['completion'][i]['content'] = clean_text(msg['content'])
                return example
            
            processed_dataset = processed_dataset.map(apply_cleaning)
        
        # Apply other preprocessing steps as needed
        
        if self.verbose:
            logger.info(f"Applied preprocessing to {stage} dataset")
        
        return processed_dataset
    
    def prepare_stage_data(self, stage: str) -> Optional[DatasetDict]:
        """
        Prepare data for a specific training stage.
        
        Args:
            stage: Training stage name
            
        Returns:
            Prepared dataset for the stage or None if no dataset available
        """
        if stage not in self.datasets:
            if self.verbose:
                logger.warning(f"No dataset specified for stage '{stage}'")
            return None
        
        # Load the dataset
        try:
            raw_dataset = self._load_dataset_from_source(self.datasets[stage])
        except Exception as e:
            logger.error(f"Failed to load dataset for stage '{stage}': {str(e)}")
            return None
        
        # Ensure it has the right splits
        dataset = self._ensure_splits(raw_dataset, stage)
        
        # Apply sample limiting
        dataset = self._limit_samples(dataset, stage)
        
        # Check required columns
        self._check_required_columns(dataset, stage)
        
        # Apply stage-specific preparation
        if stage == 'r1zero':
            prepared_dataset = self._prepare_r1zero_data(dataset)
        elif stage in ['sft', 'sft2']:
            prepared_dataset = self._prepare_sft_data(dataset, stage=stage)
        elif stage == 'reasoning_rl':
            prepared_dataset = self._prepare_reasoning_rl_data(dataset)
        elif stage == 'rejection':
            prepared_dataset = self._prepare_rejection_data(dataset)
        elif stage == 'distill':
            prepared_dataset = self._prepare_sft_data(dataset, stage='distill')
        else:
            raise ValueError(f"Unsupported training stage: {stage}")
        
        # Apply additional preprocessing
        prepared_dataset = self._apply_preprocessing(prepared_dataset, stage)
        
        self.prepared_datasets[stage] = prepared_dataset
        return prepared_dataset
    
    def prepare_data(self) -> Dict[str, DatasetDict]:
        """
        Prepare all datasets for all specified training stages.
        
        Returns:
            Dictionary mapping stage names to prepared datasets
        """
        result = {}
        
        # Prepare data for each stage
        for stage in self.datasets.keys():
            prepared_dataset = self.prepare_stage_data(stage)
            if prepared_dataset is not None:
                result[stage] = prepared_dataset
        
        # Try cold start generation if enabled and no SFT data provided
        if 'sft' not in result and self.generate_cold_start:
            cold_start_data = self._generate_cold_start_data()
            if cold_start_data is not None:
                result['sft'] = cold_start_data
        
        if not result:
            raise ValueError("No datasets were prepared. Please specify at least one dataset.")
        
        self.prepared_datasets = result
        return result
    
    def get_stage_dataloaders(self, stage: str, batch_size: int) -> Dict[str, Any]:
        """
        Get PyTorch dataloaders for a specific stage.
        
        Note: This is a placeholder for integrating with PyTorch.
        Actual implementation would depend on the training framework.
        
        Args:
            stage: Training stage name
            batch_size: Batch size for dataloaders
            
        Returns:
            Dictionary of dataloaders for each split
        """
        if stage not in self.prepared_datasets:
            if stage not in self.datasets:
                raise ValueError(f"No dataset specified for stage '{stage}'")
            self.prepare_stage_data(stage)
        
        if stage not in self.prepared_datasets:
            raise ValueError(f"Failed to prepare dataset for stage '{stage}'")
        
        dataset = self.prepared_datasets[stage]
        
        # This is just a placeholder. Would need PyTorch imports
        # and proper collate functions for actual implementation
        if self.verbose:
            logger.info(f"Returning dataloaders for stage '{stage}' with batch size {batch_size}")
        
        return {
            "train": dataset["train"],
            "validation": dataset["validation"],
            "test": dataset.get("test")
        }