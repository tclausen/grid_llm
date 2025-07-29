from world import *

def randomWalk(w, s, steps):
    values = []
    for i in range(steps):
        step = []
        v = w.look(s)
        step.append(v)
        actions = w.actionSpaceInPosition(s.position())
        a = random.choice(actions)
        step.append(a)
        s = w.step(s, a)
        values.append(step)
    return values


def randomWalkEpisodes(w, s, nSteps, nEpisodes):
    print(f"# Random walk: {nSteps} steps, {nEpisodes} episodes")
    episodes = []
    for i in range(nEpisodes):
        print(f"## episode {i} from state {s}")
        episode = randomWalk(w, s, nSteps)
        print(episode)
        episodes.append(episode)
    return episodes
