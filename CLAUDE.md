# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a grid-world reinforcement learning project with LLM integration that implements a complete pipeline to train language models on environment dynamics. The project goal is to:

1. **Train a Q-learning agent** in a grid world environment
2. **Record behavioral traces** from agent interactions 
3. **Train a GPT language model** on those traces to learn environment dynamics
4. **Use the trained LLM as a world model** for planning before taking actions

The system combines classical reinforcement learning (Q-learning) with modern language modeling to create a hybrid AI agent capable of both learning from experience and planning through language-based world simulation.

## Architecture

### Core Components

#### World System
- **`world.py`**: Grid-based environment with ASCII file-based world definitions
  - `World` class handles navigation, collision detection, and state transitions
  - `WorldComplex` subclass provides complex multi-room environments
  - World files stored in `worlds/` directory (`world_default.txt`, `world_complex.txt`)
  - 3x3 "look" function provides agent's local view as 9-character string

#### Agent System  
- **`actor.py`**: Q-learning agent with epsilon-greedy and random policies
- **`state.py`**: Agent position and world state representation
- **`qvalues.py`**: Q-value storage and retrieval for reinforcement learning
- **`point.py`**: 2D coordinate handling utilities

#### Trace Generation & Data Pipeline
- **`traceGeneration.py`**: Core trace generation system
  - `Trace` class stores sequences of (state_representation, action) pairs
  - Generates training data in format: `state|action>next_state`
  - Supports multiple trace generation with different policies
  - Vocabulary validation for tokenizer compatibility

#### Tokenization System
- **`tokenizer.py`**: Custom character-level tokenizer (19 tokens total)
  - Grid characters: ` ` (empty), `#` (wall), `*` (boundary), `A` (agent)
  - Action tokens: `u`, `d`, `l`, `r` 
  - Format tokens: `|` (separator), `>` (transition)
  - Special tokens: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`
  - Future-ready tokens: `F`, `W`, `T`, `E`, `D` for game elements

#### Dataset & Training Infrastructure
- **`dataset.py`**: PyTorch dataset classes for batched training
  - `GridTraceDataset`: Handles tokenized sequences with padding/truncation
  - `GridTraceDataLoader`: Efficient batching with proper collation
  - Automatic train/validation splitting
  
- **`train.py`**: Complete training pipeline
  - `GridWorldTrainer` class with full training loop
  - Configurable model architecture (layers, heads, embedding size)
  - Multiple optimizers and learning rate schedulers
  - Comprehensive metrics tracking (accuracy, action vs state accuracy)
  - Automatic checkpointing and model saving

#### Model System
- **`model/model.py`**: Complete GPT transformer implementation
  - Multi-head self-attention with flash attention support
  - Configurable architecture (layers, heads, embedding dimensions)
  - Support for both training and inference modes
  - Pre-trained model loading capabilities

#### Evaluation System  
- **`evaluate.py`**: Comprehensive model evaluation
  - `GridWorldEvaluator` class for systematic testing
  - State transition prediction accuracy
  - Multi-step planning capability assessment
  - Per-action performance breakdown
  - Confidence scoring for predictions

### Data Flow

1. **Data Generation**: Agent explores grid world using Q-learning or random policy
2. **Trace Recording**: System captures (state, action, next_state) transitions
3. **Tokenization**: Traces converted to token sequences using custom tokenizer
4. **Dataset Creation**: Tokenized sequences organized into training/validation sets
5. **Model Training**: GPT trained on trace data to predict state transitions
6. **Evaluation**: Trained model tested on new traces for accuracy and planning capability
7. **World Model Usage**: Trained LLM can predict environment dynamics for planning

## Development Commands

### Basic Testing
```bash
# Run main simulation and tests
python main.py

# Test individual components
python world.py              # World functionality
python actor.py              # Q-learning agent
python tokenizer.py          # Custom tokenizer
python dataset.py            # Dataset creation
python traceGeneration.py    # Trace generation
```

### Training Pipeline
```bash
# Train with default configuration
python train.py

# Train with custom config
python train.py --config config_small.json

# Train with command line overrides
python train.py --epochs 50 --batch_size 32 --lr 1e-4 --output_dir my_experiment

# Train with specific settings
python train.py --config config_small.json --epochs 100 --output_dir checkpoints_long
```

### Model Evaluation
```bash
# Evaluate trained model
python evaluate.py checkpoints/checkpoint_best.pt

# Detailed evaluation with custom parameters
python evaluate.py checkpoints/checkpoint_best.pt --test_traces 200 --plan_length 10 --samples 20

# Quick evaluation
python evaluate.py checkpoints_small/checkpoint_latest.pt --test_traces 50 --samples 5
```

### Testing and Validation
```bash
# Test model/tokenizer integration
python test_tokenizer_model.py

# Test training infrastructure  
python test_training.py

# Run comprehensive tests
python main.py  # Includes all component tests
```

## Configuration Management

### Model Configuration Files
- **`config_small.json`**: Lightweight config for quick testing
  - 4 layers, 4 heads, 256 embedding dimensions
  - 20 epochs, batch size 8, 100 traces
  - Suitable for CPU training and development

### Configuration Parameters
```json
{
  "block_size": 128,        // Maximum sequence length
  "n_layer": 4,             // Number of transformer layers  
  "n_head": 4,              // Number of attention heads
  "n_embd": 256,            // Embedding dimensions
  "dropout": 0.1,           // Dropout rate
  
  "max_epochs": 20,         // Training epochs
  "batch_size": 8,          // Batch size
  "learning_rate": 1e-3,    // Learning rate
  "scheduler": "cosine",    // LR scheduler type
  
  "num_traces": 100,        // Training traces to generate
  "steps_per_trace": 15,    // Steps per trace
  "policy": "random",       // Agent policy for data generation
  
  "output_dir": "checkpoints_small"  // Model save directory
}
```

## File Structure

### Core Implementation
```
/home/toc/tmp/grid_llm/
├── main.py                 # Main entry point and testing
├── world.py                # Grid world environment
├── actor.py                # Q-learning agent
├── state.py                # State representation
├── qvalues.py              # Q-value storage
├── point.py                # 2D coordinates
├── traceGeneration.py      # Trace recording system
├── tokenizer.py            # Custom character tokenizer
├── dataset.py              # PyTorch datasets
├── train.py                # Training infrastructure
├── evaluate.py             # Model evaluation
└── model/
    ├── model.py            # GPT transformer
    └── sample.py           # Sampling utilities
```

### Configuration & Data
```
├── config_small.json       # Small model configuration
├── pyproject.toml          # Python project config
├── worlds/
│   ├── world_default.txt   # Simple grid world
│   └── world_complex.txt   # Complex multi-room world
└── checkpoints/            # Saved model checkpoints
```

### Testing & Development
```
├── test_tokenizer_model.py # Integration tests
├── test_training.py        # Training system tests
├── log.py                  # Logging utilities
└── history.py              # Legacy trace handling
```

## Important Implementation Details

### Tokenizer Design
- **19-token vocabulary** designed specifically for grid world representation
- **Character-level tokenization** preserves spatial relationships
- **Special token support** for sequence boundaries and padding
- **Future-extensible** with reserved tokens for game elements

### Training Pipeline
- **Automatic data generation** from Q-learning agent exploration
- **Next-token prediction** training objective for sequence modeling
- **Comprehensive metrics** including action-specific accuracy
- **Flexible architecture** supporting different model sizes
- **Robust checkpointing** with best model selection

### State Representation
- **9-character strings** represent 3x3 local view around agent
- **Boundary handling** with `*` characters for world edges
- **Agent position** marked with `A` in center of view
- **Consistent format** for reliable tokenization and training

### Evaluation Framework
- **State transition accuracy** measures world model correctness
- **Planning capability** tests multi-step prediction accuracy
- **Per-action breakdown** identifies policy-specific performance
- **Confidence scoring** provides uncertainty estimates

### World Model Integration
- **Bidirectional learning**: Q-learning informs data generation, LLM provides planning
- **Hierarchical reasoning**: Local observations to global environment understanding
- **Transferable knowledge**: Trained models can generalize to similar environments

## Usage Examples

### Basic Training Workflow
```python
from train import GridWorldTrainer
from tokenizer import CharTokenizer

# Create trainer with default config
trainer = GridWorldTrainer(config)

# Train model (generates data, trains, evaluates)
model = trainer.train()
```

### Custom Data Generation
```python
from traceGeneration import generateTokenizedTraces
from world import WorldComplex

world = WorldComplex()
tokenized_data = generateTokenizedTraces(
    world, 
    num_traces=1000,
    steps_per_trace=20,
    policy="epsilon_greedy"
)
```

### Model Evaluation
```python
from evaluate import GridWorldEvaluator

evaluator = GridWorldEvaluator("checkpoints/checkpoint_best.pt")
results = evaluator.evaluate_state_transitions(test_data)
print(f"Accuracy: {results['exact_match_accuracy']:.3f}")
```

### Planning with Trained Model
```python
# Predict next state given current state and action
predicted_state, confidence = evaluator.predict_next_state(
    current_state="***# A ***", 
    action="r"
)
```

This architecture enables a complete learning loop where classical RL provides exploration data, language models learn environment dynamics, and the combination enables sophisticated planning and decision-making in grid world environments.