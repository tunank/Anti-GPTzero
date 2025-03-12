from torch.utils.data import Dataset
import torch

class TextHumanizerDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Tokenize the AI text
        input_encoding = self.tokenizer(
            item['ai_text'],
            max_length=self.max_length,
            padding='max_length', # Pad to max_length
            truncation=True, # Truncate if necessary
            return_tensors='pt' # Return PyTorch tensors
        )

        # Tokenize the human text
        target_encoding = self.tokenizer(
            item['human_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Since tokenizer return dictionary, we need to extract the input_ids and attention_mask
        input_ids = input_encoding['input_ids'].squeeze()
        attention_mask = input_encoding['attention_mask'].squeeze()
        labels = target_encoding['input_ids'].squeeze()

        # Ignore padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }