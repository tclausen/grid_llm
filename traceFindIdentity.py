#!/usr/bin/env python3

from point import *
from state import *
from log import *
from world import *
from buildTraces import *

log = getLog()

def findIdentityOps(trace, compressOp=False, unrollOp=False):
    if compressOp and unrollOp:
        log.error("compressOp and unrollOp cannot be true at the same time")
        return None
    newTrace = []
    previousState = None
    for s in trace:
        if previousState is None:
            previousState = s
            newTrace.append(s)
            continue
        if s == "u" or s == "d" or s == "l" or s == "r": # ignore actions
            newTrace.append(s)
            continue
        if s == previousState:
            if not compressOp:
                newTrace.append("I")
            else:
                newTrace.append(s)
                newTrace.append("Z")
                newTrace.append("I")
                newTrace.append(".")
            if unrollOp:
                newTrace.append("?")
                newTrace.append(s)
                newTrace.append(".")
            continue
        newTrace.append(s)
        previousState = s
    return newTrace

if __name__ == "__main__":
    w = WorldT1()
    s = State([0, 0, "r"])
    w.printAll(s)
    print(w.look(s))

    trace = randomWalk(w, s, 10)
    
    print("Trace:", trace)
    
    t = findIdentityOps(trace)
    print("Trace2:", t)

    t = findIdentityOps(trace, unrollOp=True)    
    print("Trace3:", t)

    t = findIdentityOps(trace, compressOp=True)    
    print("Trace4:", t)
