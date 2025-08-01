#!/usr/bin/env python3

import random
from world import WorldComplex
from state import State
from actor import Actor
from log import getLog
import qvalues
from tokenizer import CharTokenizer

log = getLog()

class Trace:
    """Stores a sequence of (state_representation, action) pairs from agent episodes"""
    
    def __init__(self):
        self.steps = []  # List of (state_representation, action) tuples
        
    def addStep(self, state_representation, action):
        """Add a (state_representation, action) step to the trace"""
        self.steps.append((state_representation, action))
        
    def getSteps(self):
        """Return the list of (state_representation, action) steps"""
        return self.steps
        
    def length(self):
        """Return the number of steps in the trace"""
        return len(self.steps)
        
    def __str__(self):
        return f"Trace with {len(self.steps)} steps"
        
    def __repr__(self):
        return f"Trace({self.steps})"
    
    def formatForLLM(self):
        """Format trace as LLM training data: state|action>nextstate"""
        formatted_lines = []
        
        for i in range(len(self.steps) - 1):
            current_state, action = self.steps[i]
            next_state, _ = self.steps[i + 1]
            
            # Format: state|action>nextstate
            line = f"{current_state}|{action}>{next_state}"
            formatted_lines.append(line)
        
        return formatted_lines
    
    def formatForLLMAsString(self):
        """Format trace as LLM training data and return as single string"""
        return "\n".join(self.formatForLLM())
    
    def formatForLLMTokenized(self, tokenizer=None):
        """
        Format trace as tokenized sequences for LLM training
        
        Args:
            tokenizer (CharTokenizer, optional): Tokenizer to use. Creates new one if None.
            
        Returns:
            List[List[int]]: List of tokenized sequences, one per transition
        """
        if tokenizer is None:
            tokenizer = CharTokenizer()
            
        formatted_sequences = []
        llm_formatted = self.formatForLLM()
        
        for line in llm_formatted:
            # Tokenize each line with special tokens
            tokens = tokenizer.encode_with_special_tokens(line, add_bos=True, add_eos=True)
            formatted_sequences.append(tokens)
            
        return formatted_sequences
    
    def formatForLLMTokenizedFlat(self, tokenizer=None, max_length=None):
        """
        Format trace as a single flat sequence of tokens for training
        
        Args:
            tokenizer (CharTokenizer, optional): Tokenizer to use
            max_length (int, optional): Maximum sequence length (truncate or pad)
            
        Returns:
            List[int]: Single flat sequence of tokens
        """
        if tokenizer is None:
            tokenizer = CharTokenizer()
            
        # Join all formatted lines with newlines
        full_text = self.formatForLLMAsString()
        
        # Tokenize the full sequence
        tokens = tokenizer.encode_with_special_tokens(full_text, add_bos=True, add_eos=True)
        
        # Apply max_length if specified
        if max_length is not None:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokenizer.pad_sequence(tokens, max_length)
                
        return tokens
    
    def validateVocabulary(self, tokenizer=None):
        """
        Validate that all characters in the trace are in the tokenizer vocabulary
        
        Args:
            tokenizer (CharTokenizer, optional): Tokenizer to validate against
            
        Returns:
            tuple: (is_valid: bool, unknown_chars: set)
        """
        if tokenizer is None:
            tokenizer = CharTokenizer()
            
        vocab = tokenizer.get_vocab()
        unknown_chars = set()
        
        # Check all characters in all steps
        for state_rep, action in self.steps:
            full_step = f"{state_rep}|{action}"
            for char in full_step:
                if char not in vocab:
                    unknown_chars.add(char)
        
        is_valid = len(unknown_chars) == 0
        return is_valid, unknown_chars

def generateTrace(world, start_state, num_steps, policy="random"):
    """
    Generate a trace of (state, action, next_state) transitions using Actor class
    
    Args:
        world: The world environment
        start_state: Initial state for the agent
        num_steps: Number of steps to take
        policy: Action selection policy ("random", "epsilon_greedy")
    
    Returns:
        Trace object containing the sequence of (state_representation, action) pairs
    """
    trace = Trace()
    
    # Create an Actor instance to handle stepping and action selection
    actor = Actor(world, start_state.copy())
    
    log.info(f"Generating trace with {num_steps} steps from position {actor.state.position()}")
    
    for step in range(num_steps):
        # Get current state representation (what the agent "sees")
        state_representation = actor.stateRep
        
        # Select action based on policy using Actor methods
        if policy == "random":
            action = actor.randomAction()
        elif policy == "epsilon_greedy":
            action = actor.getBestActionEpsilonGreedy(state_representation)
        else:
            raise ValueError(f"Unknown policy: {policy}")
            
        # Add step to trace (state_representation, action)
        trace.addStep(state_representation, action)
        
        # Execute action using Actor's step method
        prevStateRep = actor.stateRep
        reward = actor.step(actor.state, action)

        qvalues.add(prevStateRep, actor.stateRep, action, reward, actor.actions)
        
        log.debug(f"Step {step}: state={state_representation}, action={action}, reward={reward}")
    
    log.info(f"Generated trace with {trace.length()} steps, total reward: {actor.totalReward}")
    return trace, actor

def generateMultipleTraces(world, num_traces, steps_per_trace, policy="random"):
    """
    Generate multiple traces from random starting positions using Actor class
    
    Args:
        world: The world environment
        num_traces: Number of traces to generate
        steps_per_trace: Number of steps per trace
        policy: Action selection policy
        
    Returns:
        List of (Trace, Actor) tuples
    """
    trace_actor_pairs = []
    
    for i in range(num_traces):
        # Get a random free starting position
        start_pos = world.randomFreePos()
        start_state = State([start_pos.x, start_pos.y])
        
        log.info(f"Generating trace {i+1}/{num_traces} from position {start_pos}")
        
        trace, actor = generateTrace(world, start_state, steps_per_trace, policy)
        trace_actor_pairs.append((trace, actor))
    
    return trace_actor_pairs

def generateTokenizedTraces(world, num_traces, steps_per_trace, policy="random", tokenizer=None, max_length=None):
    """
    Generate multiple tokenized traces ready for LLM training
    
    Args:
        world: The world environment
        num_traces: Number of traces to generate
        steps_per_trace: Number of steps per trace
        policy: Action selection policy
        tokenizer: CharTokenizer instance to use
        max_length: Maximum sequence length for padding/truncation
        
    Returns:
        dict: Contains 'traces', 'actors', 'tokenized_sequences', 'validation_results'
    """
    if tokenizer is None:
        tokenizer = CharTokenizer()
    
    # Generate regular traces
    trace_actor_pairs = generateMultipleTraces(world, num_traces, steps_per_trace, policy)
    
    # Process each trace for tokenization
    tokenized_sequences = []
    validation_results = []
    
    for trace, actor in trace_actor_pairs:
        # Validate vocabulary compliance
        is_valid, unknown_chars = trace.validateVocabulary(tokenizer)
        validation_results.append((is_valid, unknown_chars))
        
        if not is_valid:
            log.warning(f"Trace contains unknown characters: {unknown_chars}")
        
        # Generate tokenized sequence
        tokenized_seq = trace.formatForLLMTokenizedFlat(tokenizer, max_length)
        tokenized_sequences.append(tokenized_seq)
    
    log.info(f"Generated {len(tokenized_sequences)} tokenized traces")
    log.info(f"Vocabulary compliance: {sum(1 for valid, _ in validation_results if valid)}/{len(validation_results)} traces valid")
    
    return {
        'traces': [trace for trace, _ in trace_actor_pairs],
        'actors': [actor for _, actor in trace_actor_pairs],
        'tokenized_sequences': tokenized_sequences,
        'validation_results': validation_results,
        'tokenizer': tokenizer
    }

if __name__ == "__main__":
    # Test trace generation with Actor
    world = WorldComplex()
    
    # Generate a single trace
    start_state = State([5, 5])
    trace, actor = generateTrace(world, start_state, 10, "epsilon_greedy")
    
    print("Generated trace:")
    for i, (state_rep, action) in enumerate(trace.getSteps()):
        print(f"Step {i}: state='{state_rep}' action='{action}'")
    
    print(f"\nTrace summary: {trace}")
    print(f"Actor summary: total reward={actor.totalReward}, avg reward={actor.avgReward():.3f}")
    print(qvalues.qvalues)
    print("Avg reward:", actor.avgReward(), actor.totalReward, actor.totalSteps)
    print(trace)
