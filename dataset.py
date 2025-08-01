#!/usr/bin/env python3

"""
Dataset classes for handling tokenized traces for LLM training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random
from typing import List, Dict, Any
from tokenizer import CharTokenizer
from log import getLog

log = getLog()

class GridTraceDataset(Dataset):
    """Dataset for tokenized grid world traces"""
    
    def __init__(self, tokenized_sequences, block_size=1024, tokenizer=None):
        """
        Initialize dataset from tokenized sequences
        
        Args:
            tokenized_sequences (List[List[int]]): List of tokenized sequences
            block_size (int): Maximum sequence length for training
            tokenizer (CharTokenizer, optional): Tokenizer instance for padding
        """
        self.tokenized_sequences = tokenized_sequences
        self.block_size = block_size
        self.tokenizer = tokenizer or CharTokenizer()
        
        # Process sequences to fit block_size
        self.processed_sequences = self._process_sequences()
        
        log.info(f"Dataset created with {len(self.processed_sequences)} sequences")
        log.info(f"Block size: {block_size}")
    
    def _process_sequences(self):
        """Process sequences to fit block_size requirements"""
        processed = []
        
        for seq in self.tokenized_sequences:
            if len(seq) <= self.block_size:
                # Pad if too short
                padded_seq = self.tokenizer.pad_sequence(seq, self.block_size)
                processed.append(padded_seq)
            else:
                # Split long sequences into multiple blocks
                for i in range(0, len(seq), self.block_size):
                    chunk = seq[i:i + self.block_size]
                    if len(chunk) == self.block_size:
                        processed.append(chunk)
                    elif len(chunk) > self.block_size // 2:  # Keep substantial chunks
                        padded_chunk = self.tokenizer.pad_sequence(chunk, self.block_size)
                        processed.append(padded_chunk)
        
        return processed
    
    def __len__(self):
        return len(self.processed_sequences)
    
    def __getitem__(self, idx):
        """
        Get a training sample
        
        Returns:
            dict: Contains 'input_ids' and 'labels' tensors
        """
        sequence = self.processed_sequences[idx]
        
        # Convert to tensor
        sequence_tensor = torch.tensor(sequence, dtype=torch.long)
        
        # Create input and target sequences for next-token prediction
        # Input: all tokens except the last
        # Target: all tokens except the first (shifted by 1)
        input_ids = sequence_tensor[:-1]  # Shape: (seq_len - 1,)
        labels = sequence_tensor[1:]      # Shape: (seq_len - 1,)
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }
    
    def get_vocab_size(self):
        """Return vocabulary size"""
        return self.tokenizer.vocab_size

class GridTraceDataLoader:
    """Wrapper for creating data loaders with proper collation"""
    
    def __init__(self, dataset, batch_size=8, shuffle=True, num_workers=0):
        """
        Initialize data loader
        
        Args:
            dataset (GridTraceDataset): Dataset instance
            batch_size (int): Batch size for training
            shuffle (bool): Whether to shuffle data
            num_workers (int): Number of worker processes
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
    
    def get_dataloader(self):
        """Create and return PyTorch DataLoader"""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=torch.cuda.is_available()
        )
    
    def _collate_fn(self, batch):
        """
        Collate function to handle batching
        
        Args:
            batch (List[dict]): List of dataset items
            
        Returns:
            dict: Batched tensors
        """
        # Stack all input_ids and labels
        input_ids = torch.stack([item['input_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,    # Shape: (batch_size, seq_len - 1)
            'labels': labels           # Shape: (batch_size, seq_len - 1)
        }

def create_train_val_datasets(tokenized_data, train_ratio=0.8, block_size=1024, tokenizer=None):
    """
    Create training and validation datasets from tokenized data
    
    Args:
        tokenized_data (dict): Output from generateTokenizedTraces
        train_ratio (float): Ratio of data for training (rest for validation)
        block_size (int): Maximum sequence length
        tokenizer (CharTokenizer, optional): Tokenizer instance
        
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    tokenized_sequences = tokenized_data['tokenized_sequences']
    
    # Shuffle sequences
    sequences_copy = tokenized_sequences.copy()
    random.shuffle(sequences_copy)
    
    # Split into train/val
    split_idx = int(len(sequences_copy) * train_ratio)
    train_sequences = sequences_copy[:split_idx]
    val_sequences = sequences_copy[split_idx:]
    
    log.info(f"Created train/val split: {len(train_sequences)} train, {len(val_sequences)} val")
    
    # Create datasets
    train_dataset = GridTraceDataset(train_sequences, block_size, tokenizer)
    val_dataset = GridTraceDataset(val_sequences, block_size, tokenizer) if val_sequences else None
    
    return train_dataset, val_dataset

def create_dataloaders(train_dataset, val_dataset=None, batch_size=8, shuffle=True):
    """
    Create training and validation data loaders
    
    Args:
        train_dataset (GridTraceDataset): Training dataset
        val_dataset (GridTraceDataset, optional): Validation dataset
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle training data
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    train_loader = GridTraceDataLoader(train_dataset, batch_size, shuffle).get_dataloader()
    val_loader = GridTraceDataLoader(val_dataset, batch_size, False).get_dataloader() if val_dataset else None
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test the dataset classes
    from traceGeneration import generateTokenizedTraces
    from world import WorldComplex
    
    print("=== Testing Dataset Classes ===")
    
    # Generate some test data
    world = WorldComplex()
    tokenized_data = generateTokenizedTraces(world, num_traces=5, steps_per_trace=10, max_length=100)
    
    print(f"Generated {len(tokenized_data['tokenized_sequences'])} tokenized sequences")
    
    # Create datasets
    train_dataset, val_dataset = create_train_val_datasets(
        tokenized_data, 
        train_ratio=0.8, 
        block_size=128
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset) if val_dataset else 0}")
    
    # Test dataset item
    sample = train_dataset[0]
    print(f"Sample input_ids shape: {sample['input_ids'].shape}")
    print(f"Sample labels shape: {sample['labels'].shape}")
    print(f"Sample input_ids: {sample['input_ids'][:20]}...")  # First 20 tokens
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, batch_size=2)
    
    # Test batch
    for batch in train_loader:
        print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"Batch labels shape: {batch['labels'].shape}")
        break
    
    print("✓ Dataset classes working correctly")