import json
import copy
from point import *


class State:
    def __init__(self, v):
        self.values = v
        self.size = len(v)

    def set(self, v):
        if len(v) != self.size:
            raise Exception("State has wrong size: " +
                            len(v) + ". Expected: " + self.size)
        self.values = v

    def setPos(self, p):
        self.values[0] = p.x
        self.values[1] = p.y
        
    def copy(self):
        v = copy.deepcopy(self.values)
        return State(v)

    def value(self, n):
        if n >= self.size or n < 0:
            raise Exception("Cannot directly get value of dimension " + str(n))
        return self.values[n]

    def setValue(self, n, v):
        if n >= self.size or n < 0:
            raise Exception("Cannot directly set value " + str(n))
        self.values[n] = v

    def position(self):
        return Point(self.values[0], self.values[1])

    def __repr__(self):
        return str(self.values)

    def __str__(self):
        return str(self.values)

    def __hash__(self):
        return hash(tuple(self.values))

    def __eq__(self, other):
        if other == None:
            return False
        return self.values == other.values

    def __ne__(self, other):
        return self.values != other.values

    def toJson(self):
        return self.values
