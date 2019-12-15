#coding: utf-8

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
import numpy as np

from src.agent import Agent
from src.environment import Environment, Status



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
    agent       = Agent(maze.shape[0] * maze.shape[1])
    environment = Environment(maze)
    episodes    = 100
    loss        = []
    start_list  = list()
    for e in range(episodes):
        if not start_list:
            start_list = environment.empty.copy()
        start_cell = random.choice(start_list)
        start_list.remove(start_cell)
        state = environment.reset(start_cell)
        score = 0
        state  = state.flatten()
        while environment.__status() == Status.PLAYING:
            #import pdb; pdb.set_trace()
            action = agent.act(state)
            next_state, reward, status = environment.step(action)
            next_state = next_state.flatten()
            done = False if status == Status.PLAYING else True
            score += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            environment.render_q(agent)
            agent.replay()
        loss.append(score)
    plt.plot([i for i in range(episodes)], loss)
    plt.xlabel("episodes")
    plt.ylabel("rewards")
    plt.show()