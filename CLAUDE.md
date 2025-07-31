# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a grid-based reinforcement learning environment with GPT language model integration. The project combines a 2D grid world simulation with Q-learning algorithms and includes a complete GPT implementation for language model tasks.

## Architecture

### Core Components

- **World System** (`world.py`): Grid-based environment with ASCII file-based world definitions stored in `worlds/`. The `World` class handles navigation, collision detection, and state transitions.

- **State Management** (`state.py`): Represents agent position and world state, integrating with the Point system for coordinate handling.

- **Actor/Agent** (`actor.py`): Implements epsilon-greedy Q-learning agent with random walk capabilities. Integrates with the qvalues system for action selection.

- **Q-Learning** (`qvalues.py`): Q-value storage and retrieval system for reinforcement learning.

- **GPT Model** (`model/model.py`): Complete GPT implementation with transformer architecture, including attention mechanisms, MLP blocks, and pre-trained model loading capabilities.

### Data Flow

1. World loads from ASCII files in `worlds/` directory
2. Actor performs actions in the world using Q-learning or random policies
3. Traces are flattened and exported for potential language model training

## Development Commands

This project uses Python with uv for dependency management:

```bash
# Run the main simulation
python main.py

# Run specific components
python world.py          # Test world functionality
python actor.py          # Test actor/Q-learning
python model/model.py    # Test GPT model
```

## Key Files

- `main.py`: Primary entry point, currently configured for basic world testing
- `pyproject.toml`: Project configuration with Python 3.12+ requirement
- `worlds/world_default.txt`: Default ASCII world definition
- `model/`: Complete GPT transformer implementation

## Important Implementation Details

- The system uses a radius-based "look" function that returns a 3x3 grid view around the agent
- World boundaries are handled with "*" characters
- Agent movement updates both world state and agent position simultaneously
- Q-learning uses epsilon-greedy exploration with configurable epsilon value
- The GPT model supports both training and inference modes with optional pre-trained weight loading