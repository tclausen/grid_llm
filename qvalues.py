import random

qvalues = {}

alpha = 0.5
gamma = 0.9

def add(s, s1, a, v, actions):
    td = alpha*(v+gamma*maxQ(s1, actions))
    #print(f"td = {td}, alpha = {alpha}, gamma = {gamma}, maxQ = {maxQ(s1, actions)}, {s}, {s1}")
    if (s, a) not in qvalues:
        qvalues[(s, a)] = td
    else:
        qvalues[(s, a)] = (1-alpha)*qvalues[(s, a)] + td

def maxQ(s, actions):
    maxQ = -100000000000000
    for a in actions:
        if (s, a) not in qvalues: 
            maxQ = max(maxQ, 0) # Unexplored actions have reward 0 by default
            continue
        if qvalues[(s, a)] > maxQ:
            maxQ = qvalues[(s, a)]
    return maxQ

def get(s, a):
    if (s, a) not in qvalues:
        return 0
    return qvalues[(s, a)]

def getBestAction(s, actions):
    bestAction = None
    bestValue = -100000000000000
    unExplored = []
    for a in actions:
        if (s, a) not in qvalues:
            #print(f"{s} {a} unexplored")
            unExplored.append(a)
            continue
        v = qvalues[(s, a)]
        if v > bestValue:
            bestValue = v
            bestAction = a
    # Unexplored actions have a 0 reward by default
    if bestValue <= 0 and len(unExplored) > 0:
        bestAction = random.choice(unExplored)
    return bestAction

def testAdd1():
    assert(qvalues == {})
    add("s1", "s2", "a1", 1, ["a1", "a2"])
    add("s2", "s3", "a2", 1, ["a1", "a2"])
    print(qvalues)
    add("s1", "s2", "a1", 1, ["a1", "a2"])
    print(qvalues)
    assert(get("s1", "a1") == 0.975)
    assert(get("s1", "a2") == 0)
    assert(get("s1", "a3") == 0)
    print(qvalues)
    print(f"Best action in s1: {getBestAction("s1", ["a1", "a2"])}")
    print(f"Best action in s2: {getBestAction("s2", ["a1", "a2"])}")
    print(f"Best action in s3 (random): {getBestAction("s3", ["a1", "a2"])}") 
    assert(getBestAction("s1", ["a1", "a2"]) == "a1")
    assert(getBestAction("s2", ["a1", "a2"]) == "a2")
    print(qvalues)

if __name__ == "__main__":
    testAdd1()