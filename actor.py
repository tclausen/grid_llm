
import qvalues
import random
import world
import state

actions = ["u", "d", "l", "r"]
epsilon = 0.05

class Actor:
    def __init__(self, w, state):
        self.world = w
        self.state = state
        self.actions = w.actionSpace()
        self.totalReward = 0
        self.totalSteps = 0
        w.setValue(state.position(), "A")

    def randomAction(self):
        #print("# Random action")
        return random.choice(actions)

    def getBestActionEpsilonGreedy(self, s):
        if random.random() < epsilon:
            return self.randomAction()
        a = qvalues.getBestAction(s, actions)
        if a:
            print(f"# Best action: {a}")
            return a
        return self.randomAction()

    def bestActionWalk(self, steps):
        n = 0
        while True:
            n += 1
            a = self.getBestActionEpsilonGreedy(s)
            s1, r = self.step(self.state, a)
            print(f"# Step {n}: {self.state}, action: {a} -> {s1}, reward: {r}")
            qvalues.add(self.state, s1, a, r, self.actions)
            if n >= steps:
                break
            self.state = s1

    def avgReward(self):
        return self.totalReward / self.totalSteps

    def step(self, s, a):
        s1, r = self.world.step(s, a)
        self.totalReward += r
        self.totalSteps += 1
        return s1, r

    def randomWalk(self, steps=10):
        n = 0
        while True:
            n += 1
            a = self.randomAction()
            s1, r = self.step(self.state, a)
            #print(f"# Step {n}: {s}, action: {a} -> {s1}, reward: {r}, {n}")
            if n >= steps:
                break
            self.state = s1

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
    a.bestActionWalk(10)
    print(qvalues.qvalues)
    print("Avg reward:", a.avgReward(), a.totalReward, a.totalSteps)
