from torch import nn
from transformers import T5ForConditionalGeneration
import torch

class TextHumanizerModel(nn.Module):
    def __init__(self, config):
        super(TextHumanizerModel, self).__init__()
        self.config = config
        # Load the pre-trained T5 model
        self.model = T5ForConditionalGeneration.from_pretrained(self.config.MODEL_NAME)

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass through the model"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs