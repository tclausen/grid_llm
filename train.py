#!/usr/bin/env python3

"""
Training script for Grid World LLM
"""

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import os
import time
import json
from datetime import datetime
import argparse

from model.model import GPT, GPTConfig
from tokenizer import CharTokenizer
from dataset import create_train_val_datasets, create_dataloaders
from traceGeneration import generateTokenizedTraces
from world import WorldComplex
from log import getLog

log = getLog()

class GridWorldTrainer:
    """Trainer class for Grid World LLM"""
    
    def __init__(self, config):
        """
        Initialize trainer
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = CharTokenizer()
        
        # Initialize model
        model_config = GPTConfig(
            vocab_size=self.tokenizer.vocab_size,
            block_size=config['block_size'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            n_embd=config['n_embd'],
            dropout=config['dropout']
        )
        
        self.model = GPT.from_scratch(model_config)
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self.model.configure_optimizers(
            weight_decay=config['weight_decay'],
            learning_rate=config['learning_rate'],
            betas=(config['beta1'], config['beta2']),
            device_type=self.device.type
        )
        
        # Initialize scheduler
        if config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=config['max_epochs'],
                eta_min=config['learning_rate'] * 0.1
            )
        elif config['scheduler'] == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=config['scheduler_step_size'],
                gamma=config['scheduler_gamma']
            )
        else:
            self.scheduler = None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Create output directory
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    def generate_training_data(self):
        """Generate training data from grid world"""
        log.info("Generating training data...")
        
        world = WorldComplex()
        tokenized_data = generateTokenizedTraces(
            world,
            num_traces=self.config['num_traces'],
            steps_per_trace=self.config['steps_per_trace'],
            policy=self.config['policy'],
            tokenizer=self.tokenizer,
            max_length=self.config['block_size']
        )
        
        # Create datasets
        train_dataset, val_dataset = create_train_val_datasets(
            tokenized_data,
            train_ratio=self.config['train_ratio'],
            block_size=self.config['block_size'],
            tokenizer=self.tokenizer
        )
        
        # Create data loaders
        train_loader, val_loader = create_dataloaders(
            train_dataset,
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        
        log.info(f"Training data: {len(train_dataset)} samples")
        log.info(f"Validation data: {len(val_dataset) if val_dataset else 0} samples")
        
        return train_loader, val_loader, tokenized_data
    
    def compute_loss(self, batch):
        """
        Compute training loss
        
        Args:
            batch (dict): Batch from data loader
            
        Returns:
            torch.Tensor: Loss value
        """
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        logits, loss = self.model(input_ids, labels)
        
        return loss
    
    def compute_metrics(self, batch, logits=None):
        """
        Compute additional metrics for evaluation
        
        Args:
            batch (dict): Batch from data loader
            logits (torch.Tensor, optional): Model logits
            
        Returns:
            dict: Dictionary of metrics
        """
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        if logits is None:
            with torch.no_grad():
                logits, _ = self.model(input_ids, labels)
        
        # Next token prediction accuracy
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == labels).float()
        accuracy = correct.mean().item()
        
        # Per-token type accuracy (actions vs states)
        action_tokens = set([4, 5, 6, 7])  # u, d, l, r
        action_mask = torch.isin(labels, torch.tensor(list(action_tokens), device=self.device))
        state_mask = ~action_mask
        
        if action_mask.any():
            action_accuracy = correct[action_mask].mean().item()
        else:
            action_accuracy = 0.0
            
        if state_mask.any():
            state_accuracy = correct[state_mask].mean().item()
        else:
            state_accuracy = 0.0
        
        return {
            'accuracy': accuracy,
            'action_accuracy': action_accuracy,
            'state_accuracy': state_accuracy,
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_metrics = {'accuracy': 0.0, 'action_accuracy': 0.0, 'state_accuracy': 0.0}
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            # Compute loss
            loss = self.compute_loss(batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Compute additional metrics
            with torch.no_grad():
                metrics = self.compute_metrics(batch)
                for key in total_metrics:
                    total_metrics[key] += metrics[key]
            
            num_batches += 1
            self.global_step += 1
            
            # Log progress
            if batch_idx % self.config['log_interval'] == 0:
                log.info(f"Epoch {self.epoch}, Batch {batch_idx}/{len(train_loader)}, "
                        f"Loss: {loss.item():.4f}, Acc: {metrics['accuracy']:.3f}")
        
        # Average metrics
        avg_loss = total_loss / num_batches
        for key in total_metrics:
            total_metrics[key] /= num_batches
        
        return avg_loss, total_metrics
    
    def validate(self, val_loader):
        """Validate on validation set"""
        if val_loader is None:
            return None, None
        
        self.model.eval()
        total_loss = 0.0
        total_metrics = {'accuracy': 0.0, 'action_accuracy': 0.0, 'state_accuracy': 0.0}
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                loss = self.compute_loss(batch)
                metrics = self.compute_metrics(batch)
                
                total_loss += loss.item()
                for key in total_metrics:
                    total_metrics[key] += metrics[key]
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        for key in total_metrics:
            total_metrics[key] /= num_batches
        
        return avg_loss, total_metrics
    
    def save_checkpoint(self, val_loss=None, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(self.output_dir, 'checkpoint_latest.pt'))
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(self.output_dir, 'checkpoint_best.pt'))
            log.info(f"Saved best checkpoint with val_loss: {val_loss:.4f}")
        
        # Save periodic checkpoint
        if self.epoch % self.config['save_interval'] == 0:
            torch.save(checkpoint, os.path.join(self.output_dir, f'checkpoint_epoch_{self.epoch}.pt'))
    
    def train(self):
        """Main training loop"""
        log.info("Starting training...")
        
        # Generate training data
        train_loader, val_loader, tokenized_data = self.generate_training_data()
        
        # Training loop
        for epoch in range(self.config['max_epochs']):
            self.epoch = epoch
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log epoch results
            log.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.3f}")
            if val_loss is not None:
                log.info(f"Epoch {epoch}: Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.3f}")
            
            # Save checkpoint
            is_best = val_loss is not None and val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(val_loss, is_best)
        
        log.info("Training completed!")
        return self.model

def get_default_config():
    """Get default training configuration"""
    return {
        # Model config
        'block_size': 256,
        'n_layer': 6,
        'n_head': 6,
        'n_embd': 384,
        'dropout': 0.1,
        
        # Training config
        'max_epochs': 100,
        'batch_size': 16,
        'learning_rate': 3e-4,
        'weight_decay': 0.1,
        'beta1': 0.9,
        'beta2': 0.95,
        'grad_clip': 1.0,
        
        # Scheduler config
        'scheduler': 'cosine',  # 'cosine', 'step', or None
        'scheduler_step_size': 30,
        'scheduler_gamma': 0.5,
        
        # Data config
        'num_traces': 1000,
        'steps_per_trace': 20,
        'policy': 'random',
        'train_ratio': 0.8,
        
        # Logging and saving
        'log_interval': 10,
        'save_interval': 10,
        'output_dir': 'checkpoints',
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Grid World LLM')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # Override with command line args
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.epochs:
        config['max_epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr
    
    # Create trainer and train
    trainer = GridWorldTrainer(config)
    model = trainer.train()