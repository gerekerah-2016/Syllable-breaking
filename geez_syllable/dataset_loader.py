"""
Custom dataset loader for Ge'ez that works with local files.
"""

import os
import json
from datasets import Dataset, DatasetDict

class GeEzDatasetLoader:
    """
    Loads Ge'ez text files and converts them to HuggingFace dataset format.
    """
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Directory containing Ge'ez text files
        """
        self.data_dir = data_dir
        self.breaker = GeEzSyllableBreaker()
    
    def load_text_files(self, split='train'):
        """
        Load all .txt files from directory.
        
        Args:
            split: Dataset split ('train', 'test', 'validation')
            
        Returns:
            List of text examples
        """
        split_dir = os.path.join(self.data_dir, split)
        texts = []
        
        if os.path.exists(split_dir):
            # Load from split directory
            for filename in os.listdir(split_dir):
                if filename.endswith('.txt'):
                    filepath = os.path.join(split_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        texts.extend([line.strip() for line in f if line.strip()])
        else:
            # Load from main directory
            for filename in os.listdir(self.data_dir):
                if filename.endswith('.txt'):
                    filepath = os.path.join(self.data_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        texts.extend([line.strip() for line in f if line.strip()])
        
        return texts
    
    def create_dataset(self, preprocess=True):
        """
        Create HuggingFace dataset from Ge'ez text files.
        
        Args:
            preprocess: If True, apply syllable breaking
            
        Returns:
            DatasetDict with train/test/validation splits
        """
        dataset_dict = {}
        
        for split in ['train', 'test', 'validation']:
            texts = self.load_text_files(split)
            
            if preprocess:
                # Apply syllable breaking
                texts = [self.breaker.break_word(text) for text in texts]
            
            if texts:  # Only create split if there's data
                dataset = Dataset.from_dict({"text": texts})
                dataset_dict[split] = dataset
        
        if not dataset_dict:
            # Create from all files if no splits found
            all_texts = self.load_text_files()
            if preprocess:
                all_texts = [self.breaker.break_word(text) for text in all_texts]
            dataset_dict['train'] = Dataset.from_dict({"text": all_texts})
        
        return DatasetDict(dataset_dict)
    
    def save_as_hf_dataset(self, output_dir: str):
        """
        Save dataset in HuggingFace format for SPLINTER.
        
        Args:
            output_dir: Directory to save dataset
        """
        dataset = self.create_dataset(preprocess=True)
        dataset.save_to_disk(output_dir)
        print(f"Saved preprocessed dataset to {output_dir}")
        
        # Also save metadata
        metadata = {
            "language": "geez",
            "script": "ethiopic",
            "preprocessing": "syllable_breaking",
            "splits": list(dataset.keys()),
            "total_examples": sum(len(dataset[split]) for split in dataset)
        }
        
        with open(os.path.join(output_dir, "dataset_info.json"), 'w') as f:
            json.dump(metadata, f, indent=2)