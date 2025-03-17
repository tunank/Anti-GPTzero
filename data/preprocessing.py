import pandas as pd
import re
import nltk
from datasets import load_dataset, Dataset
import random

nltk.download('punkt')


class DataPreprocessor:
    @staticmethod
    def load_main_dataset():
        """Load the HC3 dataset from Hugging Face"""

        dataset = load_dataset("Hello-SimpleAI/HC3", "all")
        print("Dataset loaded from Hugging Face")

        return dataset

    @staticmethod
    def create_text_pairs(dataset):
        """Extract pairs of human and AI text responses from the dataset."""
        pairs = []

        for item in dataset['train']:
            if item['human_answers'] and item['chatgpt_answers']:
                pairs.append({
                    'question': item['question'],
                    'ai_text': item['chatgpt_answers'][0],  # Take the first AI response
                    'human_text': item['human_answers'][0],  # Take the first human response
                    'source': item['source']
                })

        return pairs

    @staticmethod
    def clean_text(text):
        """Clean and normalize text by removing unwanted backslashes and extra spaces."""
        # First handle escape sequences like \n, \t, etc.
        text = re.sub(r"\\[ntr]", "", text)  # Remove common escape sequences

        # Replace escaped apostrophes with actual apostrophes
        text = re.sub(r"\\'", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @staticmethod
    def split_dataset(pairs):
        """Split dataset into train, validation, and test sets"""
        random.shuffle(pairs)

        train_size = int(len(pairs) * 0.7)
        val_size = int(len(pairs) * 0.15)

        train_pairs = pairs[:train_size]
        val_pairs = pairs[train_size:train_size + val_size]
        test_pairs = pairs[train_size + val_size:]

        print(f"Train: {len(train_pairs)}, Validation: {len(val_pairs)}, Test: {len(test_pairs)}")

        return {
            'train': train_pairs,
            'validation': val_pairs,
            'test': test_pairs
        }

    def prepare_dataset(self):
        """Main method to prepare the dataset"""
        raw_dataset = DataPreprocessor.load_main_dataset()
        pairs = DataPreprocessor.create_text_pairs(raw_dataset)
        split_data = DataPreprocessor.split_dataset(pairs)

        # Save the split data to a dictionary, and drop the column source
        hf_dataset = {
            split: Dataset.from_pandas(pd.DataFrame(data)).remove_columns('source')
            for split, data in split_data.items()
        }

        return hf_dataset


if __name__ == "__main__":
    dataset = DataPreprocessor()
    the_dataset = dataset.prepare_dataset()