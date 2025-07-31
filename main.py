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
    print("\n=== Testing Trace Generation ===")
    world = WorldComplex()
    
    # Generate a single trace
    start_state = State([5, 5])
    trace = generateTrace(world, start_state, 8)
    
    print("Generated single trace:")
    for i, (state_rep, action) in enumerate(trace.getSteps()):
        print(f"Step {i}: state='{state_rep}' action='{action}'")
    
    print(f"\nTrace summary: {trace}")
    
    # Generate multiple traces
    print("\n=== Generating Multiple Traces ===")
    traces = generateMultipleTraces(world, 3, 5)
    
    for i, trace in enumerate(traces):
        print(f"\nTrace {i+1}: {trace}")
        for j, (state_rep, action) in enumerate(trace.getSteps()):
            print(f"  Step {j}: '{state_rep}' -> '{action}'")
    
    return traces

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
