#!/usr/bin/env python3

import os
import pickle 

from state import *
from world import *
from buildTraces import *

class Node:
    def __init__(self, t=None):
        self.type = t
        self.value = None
        self.parent = None
        self.children = []

    def addChild(self, child):
        child.parent = self
        self.children.append(child)

    def isLeaf(self) -> bool:
        return len(self.children) == 0
    
    def compress(self, context, compressor):
        if self.type != "obs":
            return None
        input = context + " " + self.value + " Z"
        compressed = compressor.compress(input)
        print(f"# Compressing\n{input}")
        print(f"# Compressed\n{compressed}")
        newNode = Node("sentence")
        newNode.value = compressed
        newNode.parent = self.parent
        newNode.children.append(self)
        return newNode
    
    def __str__(self, level=0) -> str:
        ret = "   "*(level)+"-- "+str(self.type) + ": "
        if self.value:
            ret += self.value
        ret += "\n"
        for child in self.children:
            ret += child.__str__(level+1)
        return ret

class History:
    def __init__(self):
        self.root = Node("root")

    def addEpisodes(self, episodes):
        for e in episodes:
            episodeNode = Node("episode")
            self.root.addChild(episodeNode)
            for step in e:
                stepNode = Node("step")
                episodeNode.addChild(stepNode)
                obsNode = Node("obs")
                obsNode.value = step[0]
                stepNode.addChild(obsNode)
                actionNode = Node("action")
                actionNode.value = step[1]
                stepNode.addChild(actionNode)
                
    def __str__(self) -> str:
        return "History:\n" + str(self.root)
    
    def readFromFile(filename):
        f = open(filename, "rb")
        h = pickle.load(f)
        f.close()
        return h
    
    def dumpToFile(self, filename):
        f = open(filename, "wb")
        b = pickle.dump(self, f)
        f.close()

def test_dumpToFile():
    w = WorldT1()
    s = State([0, 0, "r"])
    episodes = randomWalkEpisodes(w, s, 5, 2)
    h = History()
    h.addEpisodes(episodes)
    print(h)
    h.dumpToFile("test_h1.obj")
    h2 = History.readFromFile("test_h1.obj")
    print("Loaded", h2)

if __name__ == "__main__":
    w = WorldT1()
    s = State([0, 0, "r"])
    w.printAll(s)
    print(w.look(s))
    episodes = randomWalkEpisodes(w, s, 5, 2)
    print("# Episodes:\n", episodes)

    h = History()
    h.addEpisodes(episodes)
    print(h)
    
    obsNode = h.root.children[0].children[0].children[0]
    print(obsNode)
    compressed = obsNode.compress("e e e e e e e e e d", TestCompressor())
    print(compressed)
    h.root.children[0].children[0].children[0] = compressed
    print(h)

