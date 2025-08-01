# Implementation Plan for Custom Character-Level Tokenizer

## 1. Custom Vocabulary Design

### Core Grid Characters
- ` ` (space): 0 - Empty cell
- `#`: 1 - Wall  
- `*`: 2 - Boundary/Out of bounds
- `A`: 3 - Agent position

### Action Characters  
- `u`: 4 - Up action
- `d`: 5 - Down action
- `l`: 6 - Left action
- `r`: 7 - Right action

### Format Characters
- `|`: 8 - State-action separator
- `>`: 9 - Action-nextstate separator

### Reserved Tokens for Future Features
- `F`: 10 - Food item
- `W`: 11 - Water source
- `T`: 12 - Treasure/Goal
- `E`: 13 - Enemy/Obstacle
- `D`: 14 - Door/Portal

### Special Tokens
- `<PAD>`: 15 - Padding token
- `<UNK>`: 16 - Unknown character
- `<BOS>`: 17 - Beginning of sequence
- `<EOS>`: 18 - End of sequence

**Total vocabulary size: 19**

## 2. Tokenizer Implementation

### Create `tokenizer.py`:
- `CharTokenizer` class with vocab dictionary
- `encode(text) -> List[int]` method
- `decode(tokens) -> str` method  
- `vocab_size` property
- Handle unknown characters gracefully

### Integration points:
- Replace GPT-2's tokenizer in model loading
- Update `GPTConfig.vocab_size = 19`
- Modify data preprocessing in trace generation

## 3. Model Architecture Updates

### Update `model/model.py`:
- Modify `GPTConfig` default vocab_size to 19
- Remove GPT-2 pretrained loading (incompatible vocabulary)
- Keep transformer architecture unchanged
- Update embedding layer size

### Training considerations:
- Much smaller vocabulary = faster training
- Need more training data to compensate for lack of pretraining
- Consider smaller model size (fewer parameters needed)

## 4. Data Pipeline Changes

### Update `traceGeneration.py`:
- Add tokenization step in `formatForLLM()`
- Return both raw text and tokenized sequences
- Add method `formatForLLMTokenized()` returning token IDs

### Batch processing:
- Create dataset class for loading/batching tokenized traces
- Handle sequence length padding to `block_size`
- Add data validation for vocabulary compliance

## 5. Training Infrastructure

### Training script updates:
- Custom loss calculation for next-token prediction
- Validation metrics specific to grid world (state prediction accuracy)
- Learning rate scheduling for small vocabulary
- Checkpointing and model saving

### Evaluation metrics:
- Next-state prediction accuracy
- Action prediction accuracy  
- Complete sequence generation quality
- Planning effectiveness when used in Actor

## 6. Integration with Existing System

### Actor class modifications:
- Add LLM-based planning mode
- Use trained model to simulate N-step lookahead
- Compare LLM predictions vs actual world steps
- Fallback to Q-learning if LLM confidence low

### World simulation:
- Method to validate LLM predictions against actual world
- Logging/debugging of prediction accuracy
- A/B testing framework for LLM vs Q-learning performance

## 7. Future Feature Integration

### When adding food/water/etc:
- Update world generation to include new tokens
- Extend tokenizer vocabulary (already reserved)
- Retrain model with expanded feature set
- Update Actor reward functions

### Extensibility:
- Plugin system for new token types
- Dynamic vocabulary expansion
- Transfer learning from base grid model

This plan provides a complete pathway from character-level tokenization to integrated LLM-based planning while maintaining compatibility with existing Q-learning infrastructure.