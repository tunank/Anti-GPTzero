import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os


class Trainer:
    def __init__(self, model, tokenizer, train_dataset, val_dataset, config):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config

        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Set up data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS
        )

        # Set up optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        total_steps = len(self.train_loader) * config.NUM_EPOCHS
        warmup_steps = int(total_steps * 0.1)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps, # warmup is 10% of total steps
            num_training_steps=total_steps
        )

        # Initialize tracking variables
        self.best_val_loss = float('inf')

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0

        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.MAX_GRAD_NORM)
            self.optimizer.step()
            self.scheduler.step()

            # Update progress bar
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        return epoch_loss / len(self.train_loader)

    def evaluate(self):
        """Evaluate the model on validation set"""
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Evaluating")
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                val_loss += loss.item()

                # Update progress bar
                progress_bar.set_postfix({'val_loss': loss.item()})

        return val_loss / len(self.val_loader)

    def train(self):
        """Main training loop"""
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}")

            # Train
            train_loss = self.train_epoch()

            # Evaluate
            val_loss = self.evaluate()

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(
                    f"{self.config.MODEL_SAVE_PATH}/best_model",
                    epoch,
                    train_loss,
                    val_loss
                )
                print(f"New best model saved with val_loss: {val_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % self.config.SAVE_EVERY == 0:
                self.save_model(
                    f"{self.config.MODEL_SAVE_PATH}/checkpoint_epoch_{epoch + 1}",
                    epoch,
                    train_loss,
                    val_loss
                )

    def save_model(self, path, epoch, train_loss, val_loss):
        """Save model and tokenizer"""
        if not os.path.exists(path):
            os.makedirs(path)

        # Save model
        self.model.model.save_pretrained(path)

        # Save tokenizer
        self.tokenizer.save_pretrained(path)

        # Save training state
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, os.path.join(path, "training_state.pt"))