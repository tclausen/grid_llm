#!/usr/bin/env python3

import random
from point import *
from state import *
from log import *
from world import *
from buildTraces import *

log = getLog()

idUsed = {}
sToId = {}

def getUniqueTwoLetterString(s):
    if s in sToId:
        return sToId[s]
    id = "aa"
    while id in idUsed:
        id = chr(random.randint(97, 122)) + chr(random.randint(97, 122))
    idUsed[id] = True
    sToId[s] = id
    return id

def substituteRepeats(trace, repeats):
    newTrace = []
    for s in trace:
        obs = s[0]
        a = s[1]
        if obs in repeats:
            newTrace.append([repeats[obs], a])
        else:
            newTrace.append(s)
    return newTrace

def substituteRepeatsUnrolled(trace, repeats):
    newTrace = []
    for s in trace:
        obs = s[0]
        a = s[1]
        if obs in repeats:
            newTrace.append([" ".join([repeats[obs], "?", obs, "."]), a])
        else:
            newTrace.append(s)
    return newTrace

def substituteRepeatsCompressed(trace, repeats):
    newTrace = []
    for s in trace:
        obs = s[0]
        a = s[1]
        if obs in repeats:
            newTrace.append([" ".join([obs, "Z", repeats[obs], "."]), a])
        else:
            newTrace.append(s)
    return newTrace

def findRepeats(trace):
    obss = []
    for s in trace:
        if len(s[0]) > 1:
            obs = s[0]
            obss.append(obs)
    repeats = {}
    for s in trace:
        if len(s) > 1:
            obs = s[0]
            if obs in repeats:
                continue
            if obss.count(obs) > 1:
                repeats[obs] = getUniqueTwoLetterString(obs)
                #print(s, trace.count(s))
    return repeats

if __name__ == "__main__":
    w = WorldT1()
    s = State([0, 0, "r"])
    w.printAll(s)
    print(w.look(s))

    trace = randomWalk2(w, s, 20)
    
    print("Trace:", trace)
    r = findRepeats(trace)
    print("Repeats:", r)
    
    trace2 = substituteRepeats(trace, findRepeats(trace))    
    print("Trace2:", trace2)

    trace3 = substituteRepeatsUnrolled(trace, findRepeats(trace))    
    print("Trace3:", trace3)

    trace4 = substituteRepeatsCompressed(trace, findRepeats(trace))    
    print("Trace4:", trace4)
