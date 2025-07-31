#!/usr/bin/env python3

import random
import sys
from point import *
from state import *
from log import *

log = getLog()

class World:
    def __init__(self, myid="default"):
        seed = 2
        random.seed(seed)
        log.info(f"World {str(myid)} created. Ramdom seed {seed}")
        self._data = []
        self._myid = myid
        self._readFromFile()
        self.graphs = []
        self.actions = ["u", "d", "l", "r"]

    def randomFreePos(self):
        y = random.randint(0, len(self._data)-1)
        x = random.randint(0, len(self._data[y])-1)
        p = Point(x, y)
        while not self.isFree(p):
            y = random.randint(0, len(self._data)-1)
            x = random.randint(0, len(self._data[y])-1)
            p = Point(x, y)
        return p

    def addAgent(self, p):
        self.setValue(p, "A")

    def reset(self):
        log.info("Reset world")

    def _readFromFile(self):
        raw = open("worlds/world_" + str(self._myid) + ".txt").readlines()
        self.xmax = 0
        for l in raw:
            line = l.rstrip().rstrip("|")
            if len(line) == 0:
                break
            self._data.append(list(line))
            if len(line) > self.xmax:
                self.xmax = len(line)
            # print(line)
        self.ymax = len(self._data)
        log.info("World initiated. Size: " +
                 str(self.xmax) + " x " + str(self.ymax))

    def value(self, p):
        if p.x < 0 or p.y < 0 or p.y >= len(self._data):
            return "*"
        if p.x >= len(self._data[p.y]):
            return "*"
        return self._data[p.y][p.x]

    def setValue(self, p, v):
        if p.x < 0 or p.y < 0 or p.y >= len(self._data):
            raise Exception("Point1 " + str(p) +
                            " not valid in World.setValue")
        if p.x >= len(self._data[p.y]):
            raise Exception("Point2 " + str(p) +
                            " not valid in World.setValue")
        #print(f"setvalue {p}, {v}")
        self._data[p.y][p.x] = v

    def isFree(self, p):
        v = self.value(p)
        return v == " "

    def actionSpace(self):
        return self.actions

    def actionSpaceInPosition(self, p):
        actions = []
        if self.isFree(Point(p.x+1, p.y)):
            actions.append("r")
        if self.isFree(Point(p.x, p.y+1)):
            actions.append("d")
        if self.isFree(Point(p.x-1, p.y)):
            actions.append("l")
        if self.isFree(Point(p.x, p.y-1)):
            actions.append("u")
        return actions

    def printAllCoverage(self, provider):
        y = 0
        for line in self._data:
            x = 0
            sys.stdout.write("|")
            for point in line:
                d = provider(x, y)
                if d == "":
                    d = point
                sys.stdout.write(d)
                x += 1
            print("|")
            y += 1

    def printAll(self, s = None):
        y = 0
        for line in self._data:
            x = 0
            sys.stdout.write("|")
            for point in line:
                if s and s.position() == Point(x, y):
                    sys.stdout.write("X")
                else:
                    sys.stdout.write(point)
                x += 1
            print("|")
            y += 1
        if s:
            print("State: " + str(s))

    def step(self, state, action):
        s = state.copy()
        p = s.position()
        if action == "u":
            p.y = p.y - 1
        elif action == "d":
            p.y = p.y + 1
        elif action == "l":
            p.x = p.x - 1
        elif action == "r":
            p.x = p.x + 1
        else:
            raise Exception("Unknown action:", action)
        #print(f"is free {p}, '{self.value(p)}'")
        if self.isFree(p):
            s.setPos(p)
            self.setValue(p, "A")
            self.setValue(state.position(), " ")
            return s, 0
        return s, -1
    
    def look(self, s):
        p = s.position()
        r = []
        radius = 1
        for j in range(-radius, radius+1):
            for i in range(-radius, radius+1):
                r.append(self.value(p + Point(i, j)))
            #r.append("|")
        return "".join(r)
        

class WorldT1(World):
    def __init__(self):
        super().__init__()

class WorldComplex(World):
    def __init__(self):
        super().__init__("complex")

if __name__ == "__main__":
    w = WorldT1()
    s = State([0, 0, "r"])
    w.printAll(s)
    print(w.look(s))

