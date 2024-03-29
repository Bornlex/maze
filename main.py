#coding: utf-8

import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
import numpy as np

from src.agent import Agent
from src.environment import Environment


def get_free_cell(maze, exclude=[]):
    free = []
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if maze[i, j] == 0:
                if (i, j) not in exclude:
                    free.append((i, j))
    return random.choice(free)

if __name__ == "__main__":
    maze = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 1, 1],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0]
    ])
    agent       = Agent(2)
    episodes    = 100
    loss        = []
    for e in range(episodes):
        print(f"[**] episode {e} :: {episodes}")
        start_cell = get_free_cell(maze)
        exit_cell  = get_free_cell(maze, exclude=[start_cell])
        environment = Environment(maze, start_cell=start_cell, exit_cell=exit_cell)
        environment.render(block_execution=False)
        state = environment.reset(start_cell, exit_cell)
        score = 0
        max_steps = 100
        for i in range(max_steps):
            action = agent.act(state)
            next_state, reward, done = environment.step(action)
            score += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            environment.render()
            if i % 3 == 0:
                agent.replay()
            if done:
                break
        loss.append(score)
    plt.plot([i for i in range(episodes)], loss)
    plt.xlabel("episodes")
    plt.ylabel("rewards")
    plt.show()