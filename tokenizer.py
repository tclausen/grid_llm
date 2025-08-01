#!/usr/bin/env python3

"""
Custom character-level tokenizer for grid world environment.
Maps each character to a unique token ID for LLM training.
"""

class CharTokenizer:
    """Character-level tokenizer for grid world with custom vocabulary"""
    
    def __init__(self):
        # Define vocabulary according to the plan
        self.char_to_id = {
            # Core Grid Characters
            ' ': 0,   # Empty cell
            '#': 1,   # Wall
            '*': 2,   # Boundary/Out of bounds
            'A': 3,   # Agent position
            
            # Action Characters
            'u': 4,   # Up action
            'd': 5,   # Down action
            'l': 6,   # Left action
            'r': 7,   # Right action
            
            # Format Characters
            '|': 8,   # State-action separator
            '>': 9,   # Action-nextstate separator
            
            # Reserved Tokens for Future Features
            'F': 10,  # Food item
            'W': 11,  # Water source
            'T': 12,  # Treasure/Goal
            'E': 13,  # Enemy/Obstacle
            'D': 14,  # Door/Portal
            
            # Special Tokens
            '<PAD>': 15,  # Padding token
            '<UNK>': 16,  # Unknown character
            '<BOS>': 17,  # Beginning of sequence
            '<EOS>': 18,  # End of sequence
        }
        
        # Create reverse mapping
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        
        # Special token IDs for easy access
        self.pad_token_id = 15
        self.unk_token_id = 16
        self.bos_token_id = 17
        self.eos_token_id = 18
    
    @property
    def vocab_size(self):
        """Return the vocabulary size"""
        return len(self.char_to_id)
    
    def encode(self, text):
        """
        Encode text string to list of token IDs
        
        Args:
            text (str): Input text to encode
            
        Returns:
            List[int]: List of token IDs
        """
        token_ids = []
        for char in text:
            if char in self.char_to_id:
                token_ids.append(self.char_to_id[char])
            else:
                # Handle unknown characters
                token_ids.append(self.unk_token_id)
        return token_ids
    
    def decode(self, token_ids):
        """
        Decode list of token IDs back to text string
        
        Args:
            token_ids (List[int]): List of token IDs to decode
            
        Returns:
            str: Decoded text string
        """
        chars = []
        for token_id in token_ids:
            if token_id in self.id_to_char:
                chars.append(self.id_to_char[token_id])
            else:
                # Handle unknown token IDs
                chars.append(self.id_to_char[self.unk_token_id])
        return ''.join(chars)
    
    def encode_with_special_tokens(self, text, add_bos=True, add_eos=True):
        """
        Encode text with optional BOS/EOS tokens
        
        Args:
            text (str): Input text to encode
            add_bos (bool): Whether to add beginning of sequence token
            add_eos (bool): Whether to add end of sequence token
            
        Returns:
            List[int]: List of token IDs with special tokens
        """
        token_ids = self.encode(text)
        
        if add_bos:
            token_ids = [self.bos_token_id] + token_ids
        if add_eos:
            token_ids = token_ids + [self.eos_token_id]
            
        return token_ids
    
    def pad_sequence(self, token_ids, max_length, pad_token_id=None):
        """
        Pad sequence to specified length
        
        Args:
            token_ids (List[int]): Input token sequence
            max_length (int): Target sequence length
            pad_token_id (int, optional): Padding token ID, defaults to self.pad_token_id
            
        Returns:
            List[int]: Padded sequence
        """
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
            
        if len(token_ids) >= max_length:
            return token_ids[:max_length]
        else:
            padding = [pad_token_id] * (max_length - len(token_ids))
            return token_ids + padding
    
    def get_vocab(self):
        """Return the vocabulary dictionary"""
        return self.char_to_id.copy()
    
    def __repr__(self):
        return f"CharTokenizer(vocab_size={self.vocab_size})"


if __name__ == "__main__":
    # Test the tokenizer
    tokenizer = CharTokenizer()
    
    print(f"Tokenizer: {tokenizer}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Test basic encoding/decoding
    test_text = "***# *** |u>***   ***"
    print(f"\nOriginal text: '{test_text}'")
    
    encoded = tokenizer.encode(test_text)
    print(f"Encoded: {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: '{decoded}'")
    print(f"Round-trip successful: {test_text == decoded}")
    
    # Test with special tokens
    encoded_special = tokenizer.encode_with_special_tokens(test_text)
    print(f"\nWith special tokens: {encoded_special}")
    decoded_special = tokenizer.decode(encoded_special)
    print(f"Decoded with special: '{decoded_special}'")
    
    # Test padding
    padded = tokenizer.pad_sequence(encoded, 30)
    print(f"\nPadded to 30: {padded}")
    
    # Test unknown character handling
    test_unknown = "abc#def"
    encoded_unknown = tokenizer.encode(test_unknown)
    print(f"\nUnknown chars 'abc#def': {encoded_unknown}")
    decoded_unknown = tokenizer.decode(encoded_unknown)
    print(f"Decoded unknown: '{decoded_unknown}'")