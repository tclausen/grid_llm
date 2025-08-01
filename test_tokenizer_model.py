#!/usr/bin/env python3

"""
Test script for the custom tokenizer and updated GPT model
"""

import torch
from tokenizer import CharTokenizer
from model.model import GPT, GPTConfig

def test_tokenizer():
    """Test the custom character tokenizer"""
    print("=== Testing Custom Tokenizer ===")
    
    tokenizer = CharTokenizer()
    print(f"Tokenizer: {tokenizer}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Test cases from grid world
    test_cases = [
        "***# *** |u>***   ***",  # Typical trace format
        " # |r>  #",              # Simple movement
        "A  |d> A ",              # Agent movement
        "###|l>###",              # Wall collision
        "FWT|u>FWT",              # Future features (food, water, treasure)
    ]
    
    print("\n--- Testing encode/decode ---")
    for i, text in enumerate(test_cases):
        print(f"\nTest {i+1}: '{text}'")
        
        # Basic encoding
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"  Encoded: {encoded}")
        print(f"  Decoded: '{decoded}'")
        print(f"  Round-trip OK: {text == decoded}")
        
        # With special tokens
        encoded_special = tokenizer.encode_with_special_tokens(text)
        decoded_special = tokenizer.decode(encoded_special)
        print(f"  With special tokens: {encoded_special}")
        print(f"  Decoded special: '{decoded_special}'")
    
    # Test padding
    print("\n--- Testing padding ---")
    text = "A  |r>  A"
    encoded = tokenizer.encode(text)
    padded_10 = tokenizer.pad_sequence(encoded, 10)
    padded_5 = tokenizer.pad_sequence(encoded, 5)  # Truncation test
    
    print(f"Original: {encoded} (length: {len(encoded)})")
    print(f"Padded to 10: {padded_10}")
    print(f"Truncated to 5: {padded_5}")
    
    # Test unknown characters
    print("\n--- Testing unknown characters ---")
    unknown_text = "xyz#abc"
    encoded_unknown = tokenizer.encode(unknown_text)
    decoded_unknown = tokenizer.decode(encoded_unknown)
    print(f"Unknown text: '{unknown_text}'")
    print(f"Encoded: {encoded_unknown}")
    print(f"Decoded: '{decoded_unknown}'")
    
    return tokenizer

def test_model():
    """Test the updated GPT model"""
    print("\n\n=== Testing Updated GPT Model ===")
    
    # Create model with default grid world config
    model = GPT.from_scratch()
    print(f"Model created: {model.config}")
    
    # Test with custom config
    custom_config = GPTConfig(
        vocab_size=19,
        n_layer=4,
        n_head=4,
        n_embd=256,
        block_size=128
    )
    
    custom_model = GPT.from_scratch(custom_config)
    print(f"Custom model: {custom_model.config}")
    
    # Test forward pass with dummy data
    print("\n--- Testing forward pass ---")
    batch_size = 2
    seq_length = 10
    
    # Create dummy input (random token IDs within vocab range)
    dummy_input = torch.randint(0, 19, (batch_size, seq_length))
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Dummy input: {dummy_input}")
    
    # Forward pass without targets (inference mode)
    model.eval()
    with torch.no_grad():
        logits, loss = model(dummy_input)
        print(f"Logits shape: {logits.shape}")
        print(f"Loss: {loss}")
        
        # Check output dimensions
        expected_shape = (batch_size, 1, 19)  # Last token only in inference
        assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
        print("✓ Inference forward pass successful")
    
    # Forward pass with targets (training mode)
    model.train()
    dummy_targets = torch.randint(0, 19, (batch_size, seq_length))
    logits, loss = model(dummy_input, dummy_targets)
    
    print(f"Training mode - Logits shape: {logits.shape}")
    print(f"Training mode - Loss: {loss.item():.4f}")
    
    expected_training_shape = (batch_size, seq_length, 19)
    assert logits.shape == expected_training_shape, f"Expected {expected_training_shape}, got {logits.shape}"
    print("✓ Training forward pass successful")
    
    return model

def test_integration():
    """Test tokenizer and model integration"""
    print("\n\n=== Testing Tokenizer-Model Integration ===")
    
    tokenizer = CharTokenizer()
    model = GPT.from_scratch()
    
    # Test text -> tokens -> model
    test_text = "***# *** |u>***   ***"
    print(f"Test text: '{test_text}'")
    
    # Tokenize
    tokens = tokenizer.encode_with_special_tokens(test_text, add_bos=True, add_eos=False)
    print(f"Tokens: {tokens}")
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.tensor(tokens).unsqueeze(0)  # Shape: (1, seq_len)
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, _ = model(input_tensor)
        print(f"Output logits shape: {logits.shape}")
        
        # Get probabilities and most likely next token
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        next_token_id = torch.argmax(probs).item()
        next_char = tokenizer.decode([next_token_id])
        
        print(f"Most likely next token ID: {next_token_id}")
        print(f"Most likely next character: '{next_char}'")
        print(f"Probability: {probs[next_token_id].item():.4f}")
    
    print("✓ Integration test successful")

if __name__ == "__main__":
    # Run all tests
    tokenizer = test_tokenizer()
    model = test_model()
    test_integration()
    
    print("\n" + "="*50)
    print("All tests completed successfully!")
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    print(f"Model parameters: {model.get_num_params():,}")
    print("="*50)