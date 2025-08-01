#!/usr/bin/env python3

from history import *
from world import WorldComplex
from state import State
from traceGeneration import generateTrace, generateMultipleTraces, generateTokenizedTraces
from tokenizer import CharTokenizer
from dataset import create_train_val_datasets, create_dataloaders

log = getLog()

def flattenTracesToFile(trace, filename):
    print(f"# Flattening traces to file {filename}")
    f = open(filename, "w")
    for trace in trace:
        r = flattenTrace(trace, f)
        f.write(r + "\n")
    f.close()

def flattenTrace(trace, f):
    #print("Flatten trace:", trace)
    r = " ".join(" ".join(s) for s in trace)
    return r

def testComplexWorld():
    print("=== Testing Complex World ===")
    w = WorldComplex()
    # Find a free position to start
    s = State([1, 1])  # Start at position (1,1) which should be free
    w.printAll(s)
    print('Look result:', repr(w.look(s)))
    print('World size:', w.xmax, 'x', w.ymax)
    
    # Test a few moves to see different look() results
    print("\n=== Testing different positions ===")
    positions = [[5, 5], [10, 10], [15, 15], [8, 12]]
    for pos in positions:
        s = State(pos)
        if w.isFree(s.position()):
            print(f"Position {pos}: look() = {repr(w.look(s))}")
        else:
            print(f"Position {pos}: BLOCKED")
    return w

def testTraceGeneration():
    print("\n=== Testing Trace Generation with Actor ===")
    world = WorldComplex()
    
    # Generate a single trace
    start_state = State([5, 5])
    trace, actor = generateTrace(world, start_state, 8)
    
    print("Generated single trace:")
    for i, (state_rep, action) in enumerate(trace.getSteps()):
        print(f"Step {i}: state='{state_rep}' action='{action}'")
    
    print(f"\nTrace summary: {trace}")
    print(f"Actor summary: total reward={actor.totalReward}, avg reward={actor.avgReward():.3f}")
    
    # Generate multiple traces
    print("\n=== Generating Multiple Traces ===")
    trace_actor_pairs = generateMultipleTraces(world, 3, 5)
    
    for i, (trace, actor) in enumerate(trace_actor_pairs):
        print(f"\nTrace {i+1}: {trace} (reward: {actor.totalReward})")
        for j, (state_rep, action) in enumerate(trace.getSteps()):
            print(f"  Step {j}: '{state_rep}' -> '{action}'")
    
    # Test epsilon-greedy policy
    print("\n=== Testing Epsilon-Greedy Policy ===")
    trace_eg, actor_eg = generateTrace(world, State([8, 8]), 5, policy="epsilon_greedy")
    print(f"Epsilon-greedy trace: {trace_eg} (reward: {actor_eg.totalReward})")
    
    # Test LLM formatting
    print("\n=== Testing LLM Formatting ===")
    print("LLM formatted trace:")
    llm_format = trace.formatForLLM()
    for line in llm_format:
        print(line)
    
    return trace_actor_pairs

def testTokenizedTraceGeneration():
    print("\n=== Testing Tokenized Trace Generation ===")
    world = WorldComplex()
    tokenizer = CharTokenizer()
    
    # Generate tokenized traces
    tokenized_data = generateTokenizedTraces(
        world, 
        num_traces=5, 
        steps_per_trace=8, 
        policy="random",
        tokenizer=tokenizer,
        max_length=200
    )
    
    print(f"Generated {len(tokenized_data['tokenized_sequences'])} tokenized sequences")
    
    # Show validation results
    valid_count = sum(1 for valid, _ in tokenized_data['validation_results'] if valid)
    print(f"Vocabulary compliance: {valid_count}/{len(tokenized_data['validation_results'])} traces valid")
    
    # Show sample tokenized sequence
    sample_trace = tokenized_data['traces'][0]
    sample_tokens = tokenized_data['tokenized_sequences'][0]
    
    print(f"\nSample trace ({sample_trace.length()} steps):")
    for i, (state, action) in enumerate(sample_trace.getSteps()[:3]):  # First 3 steps
        print(f"  Step {i}: '{state}' -> '{action}'")
    
    print(f"\nSample LLM format:")
    llm_formatted = sample_trace.formatForLLM()
    for line in llm_formatted[:2]:  # First 2 lines
        print(f"  {line}")
    
    print(f"\nSample tokenized sequence (length {len(sample_tokens)}):")
    print(f"  Tokens: {sample_tokens[:30]}...")  # First 30 tokens
    print(f"  Decoded: '{tokenizer.decode(sample_tokens[:30])}...'")
    
    # Test dataset creation
    print("\n=== Testing Dataset Creation ===")
    train_dataset, val_dataset = create_train_val_datasets(
        tokenized_data,
        train_ratio=0.8,
        block_size=128,
        tokenizer=tokenizer
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset) if val_dataset else 0}")
    
    # Test data loaders
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, batch_size=2)
    
    # Show a batch
    for batch in train_loader:
        print(f"Batch input shape: {batch['input_ids'].shape}")
        print(f"Batch labels shape: {batch['labels'].shape}")
        print(f"Sample input tokens: {batch['input_ids'][0][:15]}")  # First 15 tokens of first sample
        print(f"Sample label tokens: {batch['labels'][0][:15]}")
        break
    
    return tokenized_data, train_dataset, val_dataset

if __name__ == "__main__":
    # Test the complex world first
    w = testComplexWorld()
    
    # Test trace generation
    traces = testTraceGeneration()
    
    # Test tokenized trace generation and dataset creation
    tokenized_data, train_dataset, val_dataset = testTokenizedTraceGeneration()
    
    exit(0)

    episodes = randomWalkEpisodes(w, s, 5, 2)

    h = History()
    h.addEpisodes(episodes)
    h.dumpToFile("h1.obj")

    traces = []
    for e in episodes:
        traces.append(e)
        print("Episode:", e, len(e))
        repeats = findRepeats(e)
        print("Repeats:", repeats)
        trace = substituteRepeats(e, repeats)
        traces.append(trace)
        trace = substituteRepeatsUnrolled(e, repeats)
        traces.append(trace)
        trace = substituteRepeatsCompressed(e, repeats)
        traces.append(trace)

    #print("Episodes:", episodes)
    #print("Traces:", traces)
    flattenTracesToFile(traces, "input_grid.txt")
