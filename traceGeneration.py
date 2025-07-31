#!/usr/bin/env python3

import random
from world import WorldComplex
from state import State
from log import getLog

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

def generateTrace(world, start_state, num_steps, policy="random"):
    """
    Generate a trace of (state, action, next_state) transitions
    
    Args:
        world: The world environment
        start_state: Initial state for the agent
        num_steps: Number of steps to take
        policy: Action selection policy ("random" for now)
    
    Returns:
        Trace object containing the sequence of (state_representation, action) pairs
    """
    trace = Trace()
    current_state = start_state.copy()
    world.addAgent(current_state.position())
    
    log.info(f"Generating trace with {num_steps} steps from position {current_state.position()}")
    
    for step in range(num_steps):
        # Get current state representation (what the agent "sees")
        state_representation = world.look(current_state)
        
        # Get available actions at current position
        available_actions = world.actionSpaceInPosition(current_state.position())
        
        if not available_actions:
            log.warning(f"No available actions at position {current_state.position()}")
            break
            
        # Select action based on policy
        if policy == "random":
            action = random.choice(available_actions)
        else:
            raise ValueError(f"Unknown policy: {policy}")
            
        # Add step to trace (state_representation, action)
        trace.addStep(state_representation, action)
        
        # Execute action to get next state
        next_state, reward = world.step(current_state, action)
        current_state = next_state
        
        log.debug(f"Step {step}: state={state_representation}, action={action}, reward={reward}")
        world.printAll()
        print(world.look(current_state), action)
    
    log.info(f"Generated trace with {trace.length()} steps")
    return trace

def generateMultipleTraces(world, num_traces, steps_per_trace, policy="random"):
    """
    Generate multiple traces from random starting positions
    
    Args:
        world: The world environment
        num_traces: Number of traces to generate
        steps_per_trace: Number of steps per trace
        policy: Action selection policy
        
    Returns:
        List of Trace objects
    """
    traces = []
    
    for i in range(num_traces):
        # Get a random free starting position
        start_pos = world.randomFreePos()
        start_state = State([start_pos.x, start_pos.y])
        
        log.info(f"Generating trace {i+1}/{num_traces} from position {start_pos}")
        
        trace = generateTrace(world, start_state, steps_per_trace, policy)
        traces.append(trace)
    
    return traces

if __name__ == "__main__":
    # Test trace generation
    world = WorldComplex()
    
    # Generate a single trace
    start_state = State([5, 5])
    trace = generateTrace(world, start_state, 10)
    
    print("Generated trace:")
    for i, (state_rep, action) in enumerate(trace.getSteps()):
        print(f"Step {i}: state='{state_rep}' action='{action}'")
    
    print(f"\nTrace summary: {trace}")