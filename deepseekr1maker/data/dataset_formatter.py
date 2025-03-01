# deepseekr1maker/data/dataset_formatter.py
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Set
import pandas as pd
import json
import os
import yaml
import logging
import glob
import re
from datasets import Dataset, DatasetDict, load_dataset
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from tqdm import tqdm
import random
from functools import partial

logger = logging.getLogger(__name__)

class DatasetFormatter:
    """Class for formatting raw data into usable datasets for model training."""
    
    @staticmethod
    def from_csv(
        file_path: Union[str, List[str]],
        problem_column: str,
        solution_column: str,
        output_path: Optional[str] = None,
        transform_func: Optional[Callable] = None,
        output_format: str = "json",
        additional_columns: Optional[List[str]] = None,
        filter_func: Optional[Callable] = None,
        batch_size: int = 1000,
        sep: str = ",",
        encoding: str = "utf-8",
        na_values: Optional[List[str]] = None,
        verbose: bool = False
    ) -> str:
        """
        Convert CSV file(s) into a usable dataset.
        
        Args:
            file_path: Path to CSV file or list of paths
            problem_column: Name of the column containing problems
            solution_column: Name of the column containing solutions
            output_path: Path to save the formatted dataset (optional)
            transform_func: Custom transformation function (optional)
            output_format: Output format ("json", "jsonl", "parquet", "hf_dataset")
            additional_columns: List of additional columns to include
            filter_func: Function to filter data items (returns True to keep)
            batch_size: Number of rows to process at once for large files
            sep: Separator character for CSV
            encoding: File encoding
            na_values: List of strings to interpret as NA/NaN
            verbose: Whether to show progress bar
            
        Returns:
            Path to the formatted dataset
        """
        if verbose:
            logging.basicConfig(level=logging.INFO)
        
        file_paths = [file_path] if isinstance(file_path, str) else file_path
        
        # Process multiple files and combine results
        all_data = []
        total_rows = 0
        
        for path in file_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"CSV file not found: {path}")
            
            logger.info(f"Processing CSV file: {path}")
            
            # Count rows first for progress tracking
            if verbose:
                row_count = sum(1 for _ in open(path, 'r', encoding=encoding)) - 1  # -1 for header
                logger.info(f"Found {row_count} rows in {path}")
            else:
                row_count = None
            
            # Process in chunks to handle large files
            chunks = pd.read_csv(
                path, 
                chunksize=batch_size, 
                sep=sep, 
                encoding=encoding,
                na_values=na_values
            )
            
            for chunk_idx, chunk in enumerate(tqdm(chunks, desc=f"Processing {path}", disable=not verbose)):
                # Check columns
                if problem_column not in chunk.columns:
                    raise ValueError(f"Column '{problem_column}' not found in CSV.")
                if solution_column not in chunk.columns:
                    raise ValueError(f"Column '{solution_column}' not found in CSV.")
                
                # Process each row
                for _, row in chunk.iterrows():
                    # Skip rows with NaN in required columns
                    if pd.isna(row[problem_column]) or pd.isna(row[solution_column]):
                        continue
                    
                    # Create data item
                    data_item = {
                        "problem": str(row[problem_column]),
                        "solution": str(row[solution_column])
                    }
                    
                    # Add additional columns if specified
                    if additional_columns:
                        for col in additional_columns:
                            if col in chunk.columns and not pd.isna(row[col]):
                                data_item[col] = str(row[col])
                    
                    # Apply custom transformation if provided
                    if transform_func:
                        data_item = transform_func(data_item)
                        # Skip if transform returned None (for filtering)
                        if data_item is None:
                            continue
                    
                    # Apply filter if provided
                    if filter_func and not filter_func(data_item):
                        continue
                    
                    all_data.append(data_item)
                    total_rows += 1
        
        logger.info(f"Finished processing {len(file_paths)} CSV file(s). Total rows: {total_rows}")
        
        # Determine output path if not provided
        if not output_path:
            base_name = os.path.splitext(os.path.basename(file_paths[0]))[0]
            if len(file_paths) > 1:
                base_name += f"_and_{len(file_paths)-1}_more"
            
            ext = DatasetFormatter._get_extension_for_format(output_format)
            output_path = f"{base_name}_formatted{ext}"
        
        # Save formatted data
        return DatasetFormatter._save_data(all_data, output_path, output_format)
    
    @staticmethod
    def from_json(
        file_path: Union[str, List[str]],
        mapping: Dict[str, str],
        output_path: Optional[str] = None,
        transform_func: Optional[Callable] = None,
        output_format: str = "json",
        filter_func: Optional[Callable] = None,
        encoding: str = "utf-8",
        preserve_splits: bool = True,
        verbose: bool = False
    ) -> str:
        """
        Convert JSON file(s) into a usable dataset.
        
        Args:
            file_path: Path to JSON file or list of paths
            mapping: Dictionary mapping fields (e.g., {"question": "problem", "answer": "solution"})
            output_path: Path to save the formatted dataset (optional)
            transform_func: Custom transformation function (optional)
            output_format: Output format ("json", "jsonl", "parquet", "hf_dataset")
            filter_func: Function to filter data items (returns True to keep)
            encoding: File encoding
            preserve_splits: Whether to preserve train/validation/test splits if found
            verbose: Whether to show progress bar
            
        Returns:
            Path to the formatted dataset
        """
        if verbose:
            logging.basicConfig(level=logging.INFO)
        
        file_paths = [file_path] if isinstance(file_path, str) else file_path
        
        all_data_dict = {}
        has_splits = False
        
        for path in file_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"JSON file not found: {path}")
            
            logger.info(f"Processing JSON file: {path}")
            
            # Load the JSON data
            with open(path, 'r', encoding=encoding) as f:
                data = json.load(f)
            
            # Check if this is a dictionary with dataset splits
            if isinstance(data, dict) and preserve_splits:
                split_keys = ['train', 'validation', 'test', 'dev', 'val']
                if any(key in data for key in split_keys):
                    has_splits = True
                    
                    # Process each split
                    for split, split_data in data.items():
                        if not isinstance(split_data, list):
                            continue
                            
                        if split not in all_data_dict:
                            all_data_dict[split] = []
                        
                        # Process each item in the split
                        for item in tqdm(split_data, desc=f"Processing {split} split", disable=not verbose):
                            formatted_item = DatasetFormatter._map_item(item, mapping)
                            
                            # Apply custom transformation if provided
                            if transform_func:
                                formatted_item = transform_func(formatted_item)
                                # Skip if transform returned None
                                if formatted_item is None:
                                    continue
                            
                            # Apply filter if provided
                            if filter_func and not filter_func(formatted_item):
                                continue
                            
                            all_data_dict[split].append(formatted_item)
                    
                    continue
                
                # Check if it's a dictionary with a 'data' field
                if 'data' in data and isinstance(data['data'], list):
                    data = data['data']
                # Otherwise treat it as a regular item
                else:
                    data = [data]
            
            # Process as a list of items
            if not has_splits:
                if not isinstance(data, list):
                    data = [data]  # Convert single item to list
                
                if 'default' not in all_data_dict:
                    all_data_dict['default'] = []
                
                # Process each item
                for item in tqdm(data, desc=f"Processing {path}", disable=not verbose):
                    formatted_item = DatasetFormatter._map_item(item, mapping)
                    
                    # Apply custom transformation if provided
                    if transform_func:
                        formatted_item = transform_func(formatted_item)
                        # Skip if transform returned None
                        if formatted_item is None:
                            continue
                    
                    # Apply filter if provided
                    if filter_func and not filter_func(formatted_item):
                        continue
                    
                    all_data_dict['default'].append(formatted_item)
        
        # Prepare output
        if has_splits:
            logger.info(f"Finished processing with splits: {', '.join(all_data_dict.keys())}")
            for split, items in all_data_dict.items():
                logger.info(f"  {split}: {len(items)} items")
        else:
            logger.info(f"Finished processing: {len(all_data_dict.get('default', []))} total items")
        
        # Determine output path if not provided
        if not output_path:
            base_name = os.path.splitext(os.path.basename(file_paths[0]))[0]
            if len(file_paths) > 1:
                base_name += f"_and_{len(file_paths)-1}_more"
            
            ext = DatasetFormatter._get_extension_for_format(output_format)
            output_path = f"{base_name}_formatted{ext}"
        
        # Save the data
        if has_splits:
            return DatasetFormatter._save_data_with_splits(all_data_dict, output_path, output_format)
        else:
            return DatasetFormatter._save_data(all_data_dict.get('default', []), output_path, output_format)
    
    @staticmethod
    def from_jsonl(
        file_path: Union[str, List[str]],
        mapping: Dict[str, str],
        output_path: Optional[str] = None,
        transform_func: Optional[Callable] = None,
        output_format: str = "json",
        filter_func: Optional[Callable] = None,
        encoding: str = "utf-8",
        batch_size: int = 1000,
        verbose: bool = False
    ) -> str:
        """
        Convert JSONL file(s) into a usable dataset.
        
        Args:
            file_path: Path to JSONL file or list of paths
            mapping: Dictionary mapping fields
            output_path: Path to save the formatted dataset (optional)
            transform_func: Custom transformation function (optional)
            output_format: Output format ("json", "jsonl", "parquet", "hf_dataset")
            filter_func: Function to filter data items (returns True to keep)
            encoding: File encoding
            batch_size: Number of lines to process at once for large files
            verbose: Whether to show progress bar
            
        Returns:
            Path to the formatted dataset
        """
        if verbose:
            logging.basicConfig(level=logging.INFO)
        
        file_paths = [file_path] if isinstance(file_path, str) else file_path
        
        all_data = []
        total_lines = 0
        
        for path in file_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"JSONL file not found: {path}")
            
            logger.info(f"Processing JSONL file: {path}")
            
            # Count lines first for progress tracking
            if verbose:
                line_count = sum(1 for _ in open(path, 'r', encoding=encoding))
                logger.info(f"Found {line_count} lines in {path}")
            else:
                line_count = None
            
            # Process the file
            with open(path, 'r', encoding=encoding) as f:
                batch = []
                
                for line_idx, line in enumerate(tqdm(f, total=line_count, desc=f"Processing {path}", disable=not verbose)):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        item = json.loads(line)
                        formatted_item = DatasetFormatter._map_item(item, mapping)
                        
                        # Apply custom transformation if provided
                        if transform_func:
                            formatted_item = transform_func(formatted_item)
                            # Skip if transform returned None
                            if formatted_item is None:
                                continue
                        
                        # Apply filter if provided
                        if filter_func and not filter_func(formatted_item):
                            continue
                        
                        batch.append(formatted_item)
                        total_lines += 1
                        
                        # Process in batches
                        if len(batch) >= batch_size:
                            all_data.extend(batch)
                            batch = []
                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing JSON on line {line_idx + 1}: {e}")
                
                # Add the remaining batch
                if batch:
                    all_data.extend(batch)
        
        logger.info(f"Finished processing {len(file_paths)} JSONL file(s). Total items: {total_lines}")
        
        # Determine output path if not provided
        if not output_path:
            base_name = os.path.splitext(os.path.basename(file_paths[0]))[0]
            if len(file_paths) > 1:
                base_name += f"_and_{len(file_paths)-1}_more"
            
            ext = DatasetFormatter._get_extension_for_format(output_format)
            output_path = f"{base_name}_formatted{ext}"
        
        # Save formatted data
        return DatasetFormatter._save_data(all_data, output_path, output_format)
    
    @staticmethod
    def from_txt(
        file_path: Union[str, List[str]],
        delimiter: str = "###",
        problem_key: str = "problem",
        solution_key: str = "solution",
        output_path: Optional[str] = None,
        transform_func: Optional[Callable] = None,
        output_format: str = "json",
        filter_func: Optional[Callable] = None,
        encoding: str = "utf-8",
        regex_pattern: Optional[str] = None,
        verbose: bool = False
    ) -> str:
        """
        Convert TXT file(s) with delimited sections into a usable dataset.
        
        Args:
            file_path: Path to TXT file or list of paths
            delimiter: Section delimiter (default: "###")
            problem_key: Key to use for problem sections
            solution_key: Key to use for solution sections
            output_path: Path to save the formatted dataset (optional)
            transform_func: Custom transformation function (optional)
            output_format: Output format ("json", "jsonl", "parquet", "hf_dataset")
            filter_func: Function to filter data items (returns True to keep)
            encoding: File encoding
            regex_pattern: Regex pattern for advanced parsing
            verbose: Whether to show progress bar
            
        Returns:
            Path to the formatted dataset
        """
        if verbose:
            logging.basicConfig(level=logging.INFO)
        
        file_paths = [file_path] if isinstance(file_path, str) else file_path
        
        all_data = []
        total_items = 0
        
        for path in file_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Text file not found: {path}")
            
            logger.info(f"Processing text file: {path}")
            
            with open(path, 'r', encoding=encoding) as f:
                content = f.read()
            
            # Use regex pattern if provided
            if regex_pattern:
                matches = re.finditer(regex_pattern, content, re.DOTALL)
                
                for match in tqdm(list(matches), desc=f"Processing {path}", disable=not verbose):
                    groups = match.groupdict()
                    
                    if problem_key not in groups or solution_key not in groups:
                        logger.warning(f"Match does not contain required groups: {groups.keys()}")
                        continue
                    
                    data_item = {
                        "problem": groups[problem_key].strip(),
                        "solution": groups[solution_key].strip()
                    }
                    
                    # Add any additional captured groups
                    for key, value in groups.items():
                        if key not in [problem_key, solution_key]:
                            data_item[key] = value.strip()
                    
                    # Apply custom transformation if provided
                    if transform_func:
                        data_item = transform_func(data_item)
                        # Skip if transform returned None
                        if data_item is None:
                            continue
                    
                    # Apply filter if provided
                    if filter_func and not filter_func(data_item):
                        continue
                    
                    all_data.append(data_item)
                    total_items += 1
            
            # Use simple delimiter-based parsing
            else:
                sections = content.split(delimiter)
                
                # Skip the first section if it's empty
                if sections and not sections[0].strip():
                    sections = sections[1:]
                
                # Ensure even number of sections
                if len(sections) % 2 != 0:
                    logger.warning(f"Odd number of sections found, ignoring the last section.")
                    sections = sections[:-1]
                
                # Process pairs of sections
                for i in tqdm(range(0, len(sections), 2), desc=f"Processing {path}", disable=not verbose):
                    problem = sections[i].strip()
                    solution = sections[i+1].strip() if i+1 < len(sections) else ""
                    
                    data_item = {
                        "problem": problem,
                        "solution": solution
                    }
                    
                    # Apply custom transformation if provided
                    if transform_func:
                        data_item = transform_func(data_item)
                        # Skip if transform returned None
                        if data_item is None:
                            continue
                    
                    # Apply filter if provided
                    if filter_func and not filter_func(data_item):
                        continue
                    
                    all_data.append(data_item)
                    total_items += 1
        
        logger.info(f"Finished processing {len(file_paths)} text file(s). Total items: {total_items}")
        
        # Determine output path if not provided
        if not output_path:
            base_name = os.path.splitext(os.path.basename(file_paths[0]))[0]
            if len(file_paths) > 1:
                base_name += f"_and_{len(file_paths)-1}_more"
            
            ext = DatasetFormatter._get_extension_for_format(output_format)
            output_path = f"{base_name}_formatted{ext}"
        
        # Save formatted data
        return DatasetFormatter._save_data(all_data, output_path, output_format)
    
    @staticmethod
    def from_parquet(
        file_path: Union[str, List[str]],
        problem_column: str,
        solution_column: str,
        output_path: Optional[str] = None,
        transform_func: Optional[Callable] = None,
        output_format: str = "json",
        additional_columns: Optional[List[str]] = None,
        filter_func: Optional[Callable] = None,
        batch_size: int = 1000,
        verbose: bool = False
    ) -> str:
        """
        Convert Parquet file(s) into a usable dataset.
        
        Args:
            file_path: Path to Parquet file or list of paths
            problem_column: Name of the column containing problems
            solution_column: Name of the column containing solutions
            output_path: Path to save the formatted dataset (optional)
            transform_func: Custom transformation function (optional)
            output_format: Output format ("json", "jsonl", "parquet", "hf_dataset")
            additional_columns: List of additional columns to include
            filter_func: Function to filter data items (returns True to keep)
            batch_size: Number of rows to process at once for large files
            verbose: Whether to show progress bar
            
        Returns:
            Path to the formatted dataset
        """
        if verbose:
            logging.basicConfig(level=logging.INFO)
        
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow is required for Parquet support. Install with 'pip install pyarrow'.")
        
        file_paths = [file_path] if isinstance(file_path, str) else file_path
        
        all_data = []
        total_rows = 0
        
        for path in file_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Parquet file not found: {path}")
            
            logger.info(f"Processing Parquet file: {path}")
            
            # Open the Parquet file
            parquet_file = pq.ParquetFile(path)
            
            # Get schema and check columns
            schema = parquet_file.schema.names
            if problem_column not in schema:
                raise ValueError(f"Column '{problem_column}' not found in Parquet file.")
            if solution_column not in schema:
                raise ValueError(f"Column '{solution_column}' not found in Parquet file.")
            
            # Process batches
            for batch in tqdm(parquet_file.iter_batches(batch_size=batch_size), 
                             desc=f"Processing {path}", 
                             disable=not verbose,
                             total=parquet_file.num_row_groups):
                
                # Convert to pandas for easier processing
                df = batch.to_pandas()
                
                # Process each row
                for _, row in df.iterrows():
                    # Skip rows with NaN in required columns
                    if pd.isna(row[problem_column]) or pd.isna(row[solution_column]):
                        continue
                    
                    # Create data item
                    data_item = {
                        "problem": str(row[problem_column]),
                        "solution": str(row[solution_column])
                    }
                    
                    # Add additional columns if specified
                    if additional_columns:
                        for col in additional_columns:
                            if col in schema and not pd.isna(row[col]):
                                data_item[col] = str(row[col])
                    
                    # Apply custom transformation if provided
                    if transform_func:
                        data_item = transform_func(data_item)
                        # Skip if transform returned None
                        if data_item is None:
                            continue
                    
                    # Apply filter if provided
                    if filter_func and not filter_func(data_item):
                        continue
                    
                    all_data.append(data_item)
                    total_rows += 1
        
        logger.info(f"Finished processing {len(file_paths)} Parquet file(s). Total rows: {total_rows}")
        
        # Determine output path if not provided
        if not output_path:
            base_name = os.path.splitext(os.path.basename(file_paths[0]))[0]
            if len(file_paths) > 1:
                base_name += f"_and_{len(file_paths)-1}_more"
            
            ext = DatasetFormatter._get_extension_for_format(output_format)
            output_path = f"{base_name}_formatted{ext}"
        
        # Save formatted data
        return DatasetFormatter._save_data(all_data, output_path, output_format)
    
    @staticmethod
    def from_huggingface(
        dataset_name: str,
        split: Optional[str] = None,
        mapping: Dict[str, str] = None,
        output_path: Optional[str] = None,
        transform_func: Optional[Callable] = None,
        output_format: str = "json",
        filter_func: Optional[Callable] = None,
        cache_dir: Optional[str] = None,
        streaming: bool = False,
        verbose: bool = False
    ) -> str:
        """
        Convert a Hugging Face dataset into a usable format.
        
        Args:
            dataset_name: Name of the Hugging Face dataset
            split: Dataset split to use (if None, all splits are used)
            mapping: Dictionary mapping fields
            output_path: Path to save the formatted dataset (optional)
            transform_func: Custom transformation function (optional)
            output_format: Output format ("json", "jsonl", "parquet", "hf_dataset")
            filter_func: Function to filter data items (returns True to keep)
            cache_dir: Directory to use for caching
            streaming: Whether to use streaming mode for large datasets
            verbose: Whether to show progress bar
            
        Returns:
            Path to the formatted dataset
        """
        if verbose:
            logging.basicConfig(level=logging.INFO)
        
        logger.info(f"Loading Hugging Face dataset: {dataset_name}")
        
        # Load the dataset
        try:
            dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir, streaming=streaming)
        except Exception as e:
            raise ValueError(f"Error loading Hugging Face dataset '{dataset_name}': {str(e)}")
        
        # Default mapping if not provided
        if mapping is None:
            # Try to guess the mapping based on common column names
            columns = dataset.column_names
            
            question_columns = ['question', 'query', 'input', 'instruction', 'prompt']
            answer_columns = ['answer', 'response', 'output', 'completion', 'target']
            
            problem_col = None
            for col in question_columns:
                if col in columns:
                    problem_col = col
                    break
                    
            solution_col = None
            for col in answer_columns:
                if col in columns:
                    solution_col = col
                    break
            
            if problem_col is None or solution_col is None:
                raise ValueError(
                    f"Could not automatically determine mapping from columns: {columns}. "
                    f"Please provide a mapping dictionary."
                )
            
            mapping = {problem_col: "problem", solution_col: "solution"}
            logger.info(f"Using automatic mapping: {mapping}")
        
        # Check if we're dealing with a DatasetDict (multiple splits)
        if isinstance(dataset, DatasetDict):
            all_data_dict = {}
            
            for split_name, split_dataset in dataset.items():
                all_data_dict[split_name] = []
                
                for item in tqdm(split_dataset, desc=f"Processing {split_name} split", disable=not verbose):
                    formatted_item = DatasetFormatter._map_item(item, mapping)
                    
                    # Apply custom transformation if provided
                    if transform_func:
                        formatted_item = transform_func(formatted_item)
                        # Skip if transform returned None
                        if formatted_item is None:
                            continue
                    
                    # Apply filter if provided
                    if filter_func and not filter_func(formatted_item):
                        continue
                    
                    all_data_dict[split_name].append(formatted_item)
            
            logger.info(f"Finished processing with splits: {', '.join(all_data_dict.keys())}")
            for split, items in all_data_dict.items():
                logger.info(f"  {split}: {len(items)} items")
            
            # Determine output path if not provided
            if not output_path:
                # Create a safer filename from the dataset name
                safe_name = dataset_name.replace('/', '_').replace(' ', '_')
                ext = DatasetFormatter._get_extension_for_format(output_format)
                output_path = f"{safe_name}_formatted{ext}"
            
            # Save the data with splits
            return DatasetFormatter._save_data_with_splits(all_data_dict, output_path, output_format)
        
        # Single dataset
        else:
            all_data = []
            
            for item in tqdm(dataset, desc=f"Processing dataset", disable=not verbose):
                formatted_item = DatasetFormatter._map_item(item, mapping)
                
                # Apply custom transformation if provided
                if transform_func:
                    formatted_item = transform_func(formatted_item)
                    # Skip if transform returned None
                    if formatted_item is None:
                        continue
                
                # Apply filter if provided
                if filter_func and not filter_func(formatted_item):
                    continue
                
                all_data.append(formatted_item)
            
            logger.info(f"Finished processing: {len(all_data)} total items")
            
            # Determine output path if not provided
            if not output_path:
                # Create a safer filename from the dataset name
                safe_name = dataset_name.replace('/', '_').replace(' ', '_')
                if split:
                    safe_name += f"_{split}"
                ext = DatasetFormatter._get_extension_for_format(output_format)
                output_path = f"{safe_name}_formatted{ext}"
            
            # Save the data
            return DatasetFormatter._save_data(all_data, output_path, output_format)
    
    @staticmethod
    def _map_item(item: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        """Map source item fields to destination fields based on mapping."""
        formatted_item = {}
        for src_key, dst_key in mapping.items():
            if src_key in item:
                formatted_item[dst_key] = item[src_key]
        
        return formatted_item
    
    @staticmethod
    def _get_extension_for_format(output_format: str) -> str:
        """Get file extension for the specified output format."""
        format_extensions = {
            "json": ".json",
            "jsonl": ".jsonl",
            "parquet": ".parquet",
            "hf_dataset": ".arrow",
            "csv": ".csv",
            "yaml": ".yaml"
        }
        
        return format_extensions.get(output_format, ".json")
    
    @staticmethod
    def _save_data(data: List[Dict[str, Any]], output_path: str, output_format: str) -> str:
        """Save data in the specified format."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
        
        if output_format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        elif output_format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    
        elif output_format == "parquet":
            df = pd.DataFrame(data)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, output_path)
            
        elif output_format == "hf_dataset":
            dataset = Dataset.from_list(data)
            dataset.save_to_disk(output_path)
            
        elif output_format == "csv":
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False, encoding='utf-8')
            
        elif output_format == "yaml":
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f)
                
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        logger.info(f"Saved formatted data to {output_path}")
        return output_path
    
    @staticmethod
    def _save_data_with_splits(data_dict: Dict[str, List[Dict[str, Any]]], output_path: str, output_format: str) -> str:
        """Save data with splits in the specified format."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
        
        if output_format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, ensure_ascii=False, indent=2)
                
        elif output_format == "jsonl":
            # For JSONL, we create separate files for each split
            base_dir = os.path.dirname(output_path)
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            
            result_files = {}
            for split, items in data_dict.items():
                split_path = os.path.join(base_dir, f"{base_name}_{split}.jsonl")
                with open(split_path, 'w', encoding='utf-8') as f:
                    for item in items:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                result_files[split] = split_path
            
            # Create an index file
            index_path = os.path.join(base_dir, f"{base_name}_index.json")
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(result_files, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved split data to multiple files, index at {index_path}")
            return index_path
            
        elif output_format == "parquet":
            # For Parquet, we create separate files for each split
            base_dir = os.path.dirname(output_path)
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            
            result_files = {}
            for split, items in data_dict.items():
                df = pd.DataFrame(items)
                split_path = os.path.join(base_dir, f"{base_name}_{split}.parquet")
                table = pa.Table.from_pandas(df)
                pq.write_table(table, split_path)
                result_files[split] = split_path
            
            # Create an index file
            index_path = os.path.join(base_dir, f"{base_name}_index.json")
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(result_files, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved split data to multiple files, index at {index_path}")
            return index_path
            
        elif output_format == "hf_dataset":
            # For HF Dataset, we create a DatasetDict
            dataset_dict = {}
            for split, items in data_dict.items():
                dataset_dict[split] = Dataset.from_list(items)
            
            dataset = DatasetDict(dataset_dict)
            dataset.save_to_disk(output_path)
            
        elif output_format == "csv":
            # For CSV, we create separate files for each split
            base_dir = os.path.dirname(output_path)
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            
            result_files = {}
            for split, items in data_dict.items():
                df = pd.DataFrame(items)
                split_path = os.path.join(base_dir, f"{base_name}_{split}.csv")
                df.to_csv(split_path, index=False, encoding='utf-8')
                result_files[split] = split_path
            
            # Create an index file
            index_path = os.path.join(base_dir, f"{base_name}_index.json")
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(result_files, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved split data to multiple files, index at {index_path}")
            return index_path
            
        elif output_format == "yaml":
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(data_dict, f)
                
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        logger.info(f"Saved formatted data with splits to {output_path}")
        return output_path
    
    @staticmethod
    def generate_cold_start_data(
        model,
        tokenizer,
        problems: List[str],
        system_prompt: str,
        output_path: str,
        generation_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 1,
        max_new_tokens: int = 512,
        output_format: str = "json",
        device: Optional[str] = None,
        verbose: bool = False
    ) -> str:
        """
        Generate cold start data from a pretrained model.
        
        Args:
            model: Pretrained model for generating responses
            tokenizer: Tokenizer for the model
            problems: List of problems to generate responses for
            system_prompt: System prompt for generation
            output_path: Path to save the generated data
            generation_config: Configuration for generation (temperature, etc.)
            batch_size: Batch size for generation
            max_new_tokens: Maximum number of new tokens to generate
            output_format: Output format ("json", "jsonl", "parquet", "hf_dataset")
            device: Device to use for inference (default: model's current device)
            verbose: Whether to show progress bar
            
        Returns:
            Path to the generated dataset
        """
        if verbose:
            logging.basicConfig(level=logging.INFO)
        
        logger.info(f"Generating cold start data for {len(problems)} problems")
        
        # Default generation config
        default_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        }
        
        # Merge with user-provided config
        gen_config = default_config.copy()
        if generation_config:
            gen_config.update(generation_config)
        
        # Set device
        if device:
            model = model.to(device)
        
        # Generate responses
        cold_start_data = []
        
        # Process in batches
        for i in tqdm(range(0, len(problems), batch_size), desc="Generating responses", disable=not verbose):
            batch_problems = problems[i:i+batch_size]
            batch_messages = []
            
            # Prepare batch
            for problem in batch_problems:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": problem}
                ]
                batch_messages.append(messages)
            
            # Generate
            try:
                # Create inputs
                batch_inputs = []
                for messages in batch_messages:
                    input_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    batch_inputs.append(input_text)
                
                # Tokenize
                inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True).to(model.device)
                
                # Generate
                outputs = model.generate(**inputs, **gen_config)
                
                # Decode and extract responses
                for j, (problem, output) in enumerate(zip(batch_problems, outputs)):
                    # Decode
                    response = tokenizer.decode(output, skip_special_tokens=True)
                    
                    # Extract assistant response
                    assistant_response = response.split("assistant")[-1].strip()
                    
                    # Add to cold start data
                    cold_start_data.append({
                        "problem": problem,
                        "solution": assistant_response,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": problem},
                            {"role": "assistant", "content": assistant_response}
                        ]
                    })
            
            except Exception as e:
                logger.error(f"Error generating responses for batch {i//batch_size}: {str(e)}")
                # Continue with the next batch
        
        logger.info(f"Generated {len(cold_start_data)} responses")
        
        # Determine output extension
        ext = DatasetFormatter._get_extension_for_format(output_format)
        if not output_path.endswith(ext):
            output_path += ext
        
        # Save the data
        return DatasetFormatter._save_data(cold_start_data, output_path, output_format)
    
    @staticmethod
    def combine_datasets(
        file_paths: List[str],
        output_path: Optional[str] = None,
        transform_func: Optional[Callable] = None,
        output_format: str = "json",
        filter_func: Optional[Callable] = None,
        preserve_splits: bool = True,
        verbose: bool = False
    ) -> str:
        """
        Combine multiple datasets into a single dataset.
        
        Args:
            file_paths: List of paths to datasets
            output_path: Path to save the combined dataset (optional)
            transform_func: Custom transformation function (optional)
            output_format: Output format ("json", "jsonl", "parquet", "hf_dataset")
            filter_func: Function to filter data items (returns True to keep)
            preserve_splits: Whether to preserve train/validation/test splits
            verbose: Whether to show progress bar
            
        Returns:
            Path to the combined dataset
        """
        if verbose:
            logging.basicConfig(level=logging.INFO)
        
        logger.info(f"Combining {len(file_paths)} datasets")
        
        combined_data = {}
        has_splits = False
        
        for path in file_paths:
            if not os.path.exists(path):
                logger.warning(f"File not found, skipping: {path}")
                continue
            
            # Load the dataset based on file extension
            ext = os.path.splitext(path)[1].lower()
            
            try:
                if ext == '.json':
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Check if it's a dictionary with splits
                    if preserve_splits and isinstance(data, dict) and any(k in data for k in ['train', 'validation', 'test']):
                        has_splits = True
                        
                        for split, items in data.items():
                            if not isinstance(items, list):
                                continue
                                
                            if split not in combined_data:
                                combined_data[split] = []
                            
                            for item in items:
                                # Apply transformation if provided
                                if transform_func:
                                    item = transform_func(item)
                                    if item is None:
                                        continue
                                
                                # Apply filter if provided
                                if filter_func and not filter_func(item):
                                    continue
                                
                                combined_data[split].append(item)
                        
                    # Regular list of items
                    elif isinstance(data, list):
                        if 'default' not in combined_data:
                            combined_data['default'] = []
                        
                        for item in data:
                            # Apply transformation if provided
                            if transform_func:
                                item = transform_func(item)
                                if item is None:
                                    continue
                            
                            # Apply filter if provided
                            if filter_func and not filter_func(item):
                                continue
                            
                            combined_data['default'].append(item)
                    
                elif ext == '.jsonl':
                    if 'default' not in combined_data:
                        combined_data['default'] = []
                    
                    with open(path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            
                            item = json.loads(line)
                            
                            # Apply transformation if provided
                            if transform_func:
                                item = transform_func(item)
                                if item is None:
                                    continue
                            
                            # Apply filter if provided
                            if filter_func and not filter_func(item):
                                continue
                            
                            combined_data['default'].append(item)
                
                elif ext == '.parquet':
                    if 'default' not in combined_data:
                        combined_data['default'] = []
                    
                    table = pq.read_table(path)
                    df = table.to_pandas()
                    
                    for _, row in df.iterrows():
                        item = row.to_dict()
                        
                        # Apply transformation if provided
                        if transform_func:
                            item = transform_func(item)
                            if item is None:
                                continue
                        
                        # Apply filter if provided
                        if filter_func and not filter_func(item):
                            continue
                        
                        combined_data['default'].append(item)
                
                elif ext in ['.arrow', '']:
                    # Try to load as a Hugging Face dataset
                    try:
                        dataset = load_dataset(path)
                        
                        if isinstance(dataset, DatasetDict):
                            has_splits = True
                            
                            for split, split_dataset in dataset.items():
                                if split not in combined_data:
                                    combined_data[split] = []
                                
                                for item in split_dataset:
                                    # Apply transformation if provided
                                    if transform_func:
                                        item = transform_func(item)
                                        if item is None:
                                            continue
                                    
                                    # Apply filter if provided
                                    if filter_func and not filter_func(item):
                                        continue
                                    
                                    combined_data[split].append(item)
                        
                        else:
                            if 'default' not in combined_data:
                                combined_data['default'] = []
                            
                            for item in dataset:
                                # Apply transformation if provided
                                if transform_func:
                                    item = transform_func(item)
                                    if item is None:
                                        continue
                                
                                # Apply filter if provided
                                if filter_func and not filter_func(item):
                                    continue
                                
                                combined_data['default'].append(item)
                                
                    except Exception as e:
                        logger.warning(f"Error loading as HF dataset: {str(e)}")
                
                else:
                    logger.warning(f"Unsupported file format: {ext}, skipping: {path}")
            
            except Exception as e:
                logger.warning(f"Error processing file {path}: {str(e)}")
        
        # Determine output path if not provided
        if not output_path:
            output_format_ext = DatasetFormatter._get_extension_for_format(output_format)
            output_path = f"combined_dataset{output_format_ext}"
        
        # Log dataset sizes
        if has_splits:
            logger.info(f"Combined dataset with splits:")
            for split, items in combined_data.items():
                logger.info(f"  {split}: {len(items)} items")
        else:
            logger.info(f"Combined dataset: {len(combined_data.get('default', []))} items")
        
        # Save the combined dataset
        if has_splits:
            return DatasetFormatter._save_data_with_splits(combined_data, output_path, output_format)
        else:
            return DatasetFormatter._save_data(combined_data.get('default', []), output_path, output_format)
    
    @staticmethod
    def convert_format(
        input_path: str,
        output_path: str,
        output_format: str,
        transform_func: Optional[Callable] = None,
        filter_func: Optional[Callable] = None,
        verbose: bool = False
    ) -> str:
        """
        Convert a dataset from one format to another.
        
        Args:
            input_path: Path to the input dataset
            output_path: Path to save the converted dataset
            output_format: Output format ("json", "jsonl", "parquet", "hf_dataset")
            transform_func: Custom transformation function (optional)
            filter_func: Function to filter data items (returns True to keep)
            verbose: Whether to show progress bar
            
        Returns:
            Path to the converted dataset
        """
        if verbose:
            logging.basicConfig(level=logging.INFO)
        
        logger.info(f"Converting {input_path} to {output_format} format")
        
        # Determine input format
        input_ext = os.path.splitext(input_path)[1].lower()
        
        # Load the dataset
        try:
            if input_ext == '.json':
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check if it's a dictionary with splits
                if isinstance(data, dict) and any(k in data for k in ['train', 'validation', 'test']):
                    # Process with splits
                    processed_data = {}
                    
                    for split, items in data.items():
                        if not isinstance(items, list):
                            continue
                            
                        processed_data[split] = []
                        
                        for item in tqdm(items, desc=f"Processing {split}", disable=not verbose):
                            # Apply transformation if provided
                            if transform_func:
                                item = transform_func(item)
                                if item is None:
                                    continue
                            
                            # Apply filter if provided
                            if filter_func and not filter_func(item):
                                continue
                            
                            processed_data[split].append(item)
                    
                    # Save with splits
                    return DatasetFormatter._save_data_with_splits(processed_data, output_path, output_format)
                
                # Regular list of items
                elif isinstance(data, list):
                    processed_data = []
                    
                    for item in tqdm(data, desc="Processing", disable=not verbose):
                        # Apply transformation if provided
                        if transform_func:
                            item = transform_func(item)
                            if item is None:
                                continue
                        
                        # Apply filter if provided
                        if filter_func and not filter_func(item):
                            continue
                        
                        processed_data.append(item)
                    
                    # Save normally
                    return DatasetFormatter._save_data(processed_data, output_path, output_format)
                
                else:
                    raise ValueError(f"Unsupported JSON structure in {input_path}")
            
            elif input_ext == '.jsonl':
                processed_data = []
                
                with open(input_path, 'r', encoding='utf-8') as f:
                    for line in tqdm(f, desc="Processing", disable=not verbose):
                        line = line.strip()
                        if not line:
                            continue
                        
                        item = json.loads(line)
                        
                        # Apply transformation if provided
                        if transform_func:
                            item = transform_func(item)
                            if item is None:
                                continue
                        
                        # Apply filter if provided
                        if filter_func and not filter_func(item):
                            continue
                        
                        processed_data.append(item)
                
                # Save normally
                return DatasetFormatter._save_data(processed_data, output_path, output_format)
            
            elif input_ext == '.parquet':
                table = pq.read_table(input_path)
                df = table.to_pandas()
                
                processed_data = []
                
                for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing", disable=not verbose):
                    item = row.to_dict()
                    
                    # Apply transformation if provided
                    if transform_func:
                        item = transform_func(item)
                        if item is None:
                            continue
                    
                    # Apply filter if provided
                    if filter_func and not filter_func(item):
                        continue
                    
                    processed_data.append(item)
                
                # Save normally
                return DatasetFormatter._save_data(processed_data, output_path, output_format)
            
            elif input_ext == '.csv':
                df = pd.read_csv(input_path)
                
                processed_data = []
                
                for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing", disable=not verbose):
                    item = row.to_dict()
                    
                    # Apply transformation if provided
                    if transform_func:
                        item = transform_func(item)
                        if item is None:
                            continue
                    
                    # Apply filter if provided
                    if filter_func and not filter_func(item):
                        continue
                    
                    processed_data.append(item)
                
                # Save normally
                return DatasetFormatter._save_data(processed_data, output_path, output_format)
            
            else:
                # Try to load as a Hugging Face dataset
                try:
                    dataset = load_dataset(input_path)
                    
                    if isinstance(dataset, DatasetDict):
                        # Process with splits
                        processed_data = {}
                        
                        for split, split_dataset in dataset.items():
                            processed_data[split] = []
                            
                            for item in tqdm(split_dataset, desc=f"Processing {split}", disable=not verbose):
                                # Apply transformation if provided
                                if transform_func:
                                    item = transform_func(item)
                                    if item is None:
                                        continue
                                
                                # Apply filter if provided
                                if filter_func and not filter_func(item):
                                    continue
                                
                                processed_data[split].append(item)
                        
                        # Save with splits
                        return DatasetFormatter._save_data_with_splits(processed_data, output_path, output_format)
                    
                    else:
                        processed_data = []
                        
                        for item in tqdm(dataset, desc="Processing", disable=not verbose):
                            # Apply transformation if provided
                            if transform_func:
                                item = transform_func(item)
                                if item is None:
                                    continue
                            
                            # Apply filter if provided
                            if filter_func and not filter_func(item):
                                continue
                            
                            processed_data.append(item)
                        
                        # Save normally
                        return DatasetFormatter._save_data(processed_data, output_path, output_format)
                
                except Exception as e:
                    raise ValueError(f"Unsupported file format or error: {str(e)}")
        
        except Exception as e:
            raise RuntimeError(f"Error converting dataset: {str(e)}")