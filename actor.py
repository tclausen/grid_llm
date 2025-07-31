
import qvalues
import random
import world
import state

actions = ["u", "d", "l", "r"]
epsilon = 0.1

class Actor:
    def __init__(self, w, state):
        self.world = w
        w.addAgent(state.position())
        self.state = state
        self.stateRep = w.look(state)
        self.actions = w.actionSpace()
        self.totalReward = 0
        self.totalSteps = 0

    def randomAction(self):
        #print("# Random action")
        return random.choice(actions)

    def getBestActionEpsilonGreedy(self, s):
        if random.random() < epsilon:
            return self.randomAction()
        a = qvalues.getBestAction(s, actions)
        if a:
            #print(f"# Best action: {a}")
            return a
        return self.randomAction()

    def bestActionWalk(self, steps):
        n = 0
        while True:
            n += 1
            a = self.getBestActionEpsilonGreedy(self.stateRep)
            prevState = self.state
            prevStateRep = self.stateRep
            r = self.step(self.state, a)
            print(f"# Step {n}: {prevState}, action: {a} -> {self.state}, reward: {r}")
            qvalues.add(prevStateRep, self.stateRep, a, r, self.actions)
            if n >= steps:
                break

    def avgReward(self):
        return self.totalReward / self.totalSteps

    def step(self, s, a):
        self.state, r = self.world.step(self.state, a)
        self.stateRep = self.world.look(self.state)
        self.totalReward += r
        self.totalSteps += 1
        return r

    def randomWalk(self, steps=10):
        n = 0
        while True:
            n += 1
            a = self.randomAction()
            r = self.step(self.state, a)
            #print(f"# Step {n}: {s}, action: {a} -> {s1}, reward: {r}, {n}")
            if n >= steps:
                break

if __name__ == "__main__":
    w = world.WorldT1()
    s = state.State([0, 0])
    a = Actor(w, s)
    a.randomWalk(10)
    print(qvalues.qvalues)
    print("Avg reward:", a.avgReward(), a.totalReward, a.totalSteps)

    w = world.WorldT1()
    s = state.State([0, 0])
    a = Actor(w, s)
    a.bestActionWalk(10000)
    print(qvalues.qvalues)
    print("Avg reward:", a.avgReward(), a.totalReward, a.totalSteps)
