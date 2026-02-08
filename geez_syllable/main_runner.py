#!/usr/bin/env python3
"""
Main script to run SPLINTER with Ge'ez Syllable Breaking.
Uses original SPLINTER code without modifications.
"""

import os
import sys
import json

# Add both paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # Current dir
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))  # SPLINTER

def main():
    print("=" * 70)
    print("SPLINTER with Ge'ez Syllable Breaking")
    print("=" * 70)
    
    # Import your syllable breaking module
    from geez_syllable import GeEzSyllableBreaker, GeEzLanguageAdapter, GeEzDatasetLoader
    
    # Step 1: Preprocess your dataset
    print("\n📚 Step 1: Loading and preprocessing Ge'ez dataset...")
    
    # Load your Ge'ez data
    data_loader = GeEzDatasetLoader("Geez-Dataset")
    
    # Create and save as HF dataset
    output_dir = "data/processed_geez"
    os.makedirs(output_dir, exist_ok=True)
    data_loader.save_as_hf_dataset(output_dir)
    
    # Step 2: Patch SPLINTER to use your language adapter
    print("\n🔧 Step 2: Configuring SPLINTER for Ge'ez...")
    
    # Create a modified params.py in memory
    geez_config = {
        "LANGUAGE": "geez",
        "SPLINTER_TRAINING_CORPUS_PATH": output_dir,  # Local processed dataset
        "SPLINTER_TRAINING_CORPUS_NAME": "train",      # Split name
        "SPLINTER_LETTERS_SUBSET": None,
        "IS_ENCODED": True,
        "SAVE_CORPORA_INTO_FILE": True,
        "TRAIN_TOKENIZERS": True,
        "TOKENIZE_CORPORA": True,
        "RUN_STATIC_CHECKS": True,
        "TOKENIZERS_TYPES": ["unigram", "bpe"],
        "TOKENIZERS_VOCAB_SIZES": [1000, 2000, 5000],
        "STATIC_CHECKS_CORPORA": ["geez_corpus"],
        "TASK_ID": 999  # Unique ID for Ge'ez
    }
    
    # Save config
    config_path = "geez_config.json"
    with open(config_path, 'w') as f:
        json.dump(geez_config, f, indent=2)
    
    # Step 3: Monkey-patch SPLINTER's factory
    print("\n⚡ Step 3: Integrating with SPLINTER...")
    
    # Import SPLINTER components
    from src.SplinterTrainer import SplinterTrainer
    from src.language_utils.LanguageUtilsFactory import LanguageUtilsFactory
    
    # Create your language adapter
    geez_adapter = GeEzLanguageAdapter()
    
    # Monkey-patch the factory to return our adapter for 'geez'
    original_get_by_language = LanguageUtilsFactory.get_by_language
    
    def patched_get_by_language(language: str):
        if language == "geez":
            return geez_adapter
        return original_get_by_language(language)
    
    LanguageUtilsFactory.get_by_language = staticmethod(patched_get_by_language)
    
    # Step 4: Run SPLINTER training
    print("\n🚀 Step 4: Running SPLINTER training with syllable breaking...")
    
    try:
        # Create trainer
        trainer = SplinterTrainer(geez_adapter)
        
        # Train with your processed dataset
        print(f"Training with dataset: {output_dir}")
        
        # Note: We need to modify get_word_dict to accept local datasets
        # Let's patch it temporarily
        import src.SplinterTrainer as st_module
        
        original_get_word_dict = trainer.get_word_dict
        
        def patched_get_word_dict(dataset_path, dataset_name):
            """Handle local datasets."""
            if os.path.exists(dataset_path):
                # It's a local directory
                print(f"Loading local dataset from {dataset_path}")
                
                # Create simple dataset
                from datasets import Dataset
                import glob
                
                # Find all text files
                txt_files = glob.glob(os.path.join(dataset_path, "**/*.txt"), recursive=True)
                if not txt_files:
                    txt_files = glob.glob(os.path.join(dataset_path, "*.txt"))
                
                texts = []
                for txt_file in txt_files:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        texts.extend(f.read().splitlines())
                
                # Create dataset
                dataset = Dataset.from_dict({"text": texts})
                
                # Extract words
                from src.CorpusWordsExtractor import CorpusWordsExtractor
                extractor = CorpusWordsExtractor(geez_adapter)
                
                # Create words dict
                words_dict = {}
                for example in dataset:
                    for word in example['text'].split():
                        if geez_adapter.is_word_contains_letters_from_other_languages(word):
                            continue
                        if word in words_dict:
                            words_dict[word] += 1
                        else:
                            words_dict[word] = 1
                
                return words_dict
            else:
                # Use original method
                return original_get_word_dict(dataset_path, dataset_name)
        
        trainer.get_word_dict = patched_get_word_dict.__get__(trainer, SplinterTrainer)
        
        # Run training
        reductions, char_map, inv_map = trainer.train(
            dataset_path=output_dir,
            dataset_name="train",
            letters_for_reductions=None
        )
        
        print("\n✅ SUCCESS! Ge'ez Syllable Breaking completed!")
        print(f"\n📊 Results:")
        print(f"  - Generated {len(reductions)} reduction rules")
        print(f"  - Created {len(char_map)} new character mappings")
        print(f"\n💾 Output saved to:")
        print(f"  - data/splinter/reductions_map.json")
        print(f"  - data/splinter/new_unicode_chars.json")
        
        # Show example
        print("\n📝 Example transformation:")
        test_word = "ባሕር"
        broken = geez_adapter.breaker.break_word(test_word)
        restored = geez_adapter.breaker.restore_word(broken)
        print(f"  Original: {test_word}")
        print(f"  Syllable Broken: {broken}")
        print(f"  Restored: {restored}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n💡 Tips:")
        print("1. Make sure 'Geez-Dataset' folder contains .txt files")
        print("2. Check that all SPLINTER dependencies are installed")
        print("3. Try running with a small dataset first")

if __name__ == "__main__":
    main()