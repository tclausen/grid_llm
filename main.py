#!/usr/bin/env python3

from history import *
from world import WorldComplex
from state import State
from traceGeneration import generateTrace, generateMultipleTraces

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

if __name__ == "__main__":
    # Test the complex world first
    w = testComplexWorld()
    
    # Test trace generation
    traces = testTraceGeneration()
    
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
