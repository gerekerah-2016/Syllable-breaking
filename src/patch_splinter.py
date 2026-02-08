"""
Simple patch to make SPLINTER load local text files.
"""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

# Monkey-patch the load_dataset function
from datasets import load_dataset as original_load_dataset

def patched_load_dataset(dataset_path, dataset_name, split="train", cache_dir=None):
    """Handle local text files."""
    print(f"Trying to load: path={dataset_path}, name={dataset_name}")
    
    # Check if it's a local text file
    if dataset_path == "text" and os.path.exists(dataset_name):
        print(f"Loading local file: {dataset_name}")
        return original_load_dataset("text", data_files=dataset_name, split=split, cache_dir=cache_dir)
    
    # Try with relative path
    if os.path.exists(dataset_name):
        print(f"Loading local file (relative): {dataset_name}")
        return original_load_dataset("text", data_files=dataset_name, split=split, cache_dir=cache_dir)
    
    # Original behavior
    return original_load_dataset(dataset_path, dataset_name, split=split, cache_dir=cache_dir)

# Apply patch
import src.SplinterTrainer
import datasets
datasets.load_dataset = patched_load_dataset
src.SplinterTrainer.load_dataset = patched_load_dataset

print("Patched load_dataset to handle local files!")