#!/usr/bin/env python3

from traceFindRepeats import *
from buildTraces import *
from history import *

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

if __name__ == "__main__":
    w = WorldT1()
    s = State([0, 0])
    w.printAll(s)
    print(w.look(s))

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
