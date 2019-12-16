#coding: utf-8

import matplotlib.pyplot as plt
import numpy as np


class Environment(object):
    def __init__(self, maze, start_cell, exit_cell):
        """
        0: move left
        1: move right
        2: move up
        3: move down
        """
        self._actions = [0, 1, 2, 3]
        self._maze    = maze
        self._axis    = None

        self._start_cell  = start_cell
        self._exit_cell   = exit_cell
        self._current     = start_cell
        
        self._exit_reward = 10
        self._move_reward = -0.01
        self._visited_reward = -0.5
        self._illegal_reward = -2

        self._visited_cells = []
    
    def reset(self, start_cell, exit_cell):
        self._visited_cells = []
        self._start_cell    = start_cell
        self._exit_cell     = exit_cell
        self._current       = start_cell
        col, row = self._current
        return np.array([row, col])
    
    def _draw(self):
        self._axis.plot(*self._current, "ro")
        self._axis.get_figure().canvas.draw()
        self._axis.get_figure().canvas.flush_events()
    
    def _get_legal_actions(self):
        actions = []
        col, row = self._current
        if col - 1 >= 0 and self._maze[row, col - 1] == 0:
            actions.append(0)
        if col + 1 <= self._maze.shape[0] - 1 and self._maze[row, col + 1] == 0:
            actions.append(1)
        if row - 1 >= 0 and self._maze[row - 1, col] == 0:
            actions.append(2)
        if row + 1 <= self._maze.shape[1] - 1 and self._maze[row + 1, col] == 0:
            actions.append(3)
        return actions

    def step(self, action):
        legal_actions = self._get_legal_actions()
        done = False
        col, row = self._current
        if action not in legal_actions:
            reward = self._illegal_reward
        else:
            if action == 0:
                col -= 1
            elif action == 1:
                col += 1
            elif action == 2:
                row -= 1
            elif action == 3:
                row += 1
            self._current = (col, row)
            reward = 0
            if self._current == self._exit_cell:
                reward = self._exit_reward
                done = True
            elif self._current in self._visited_cells:
                reward = self._visited_reward
            else:
                reward = self._move_reward
            self._visited_cells.append(self._current)
        self._draw()
        col, row = self._current
        state = np.array([row, col])
        return state, reward, done
    
    def _render(self):
        nrows, ncols = self._maze.shape
        self._axis.clear()
        self._axis.set_xticks(np.arange(0.5, nrows, step=1))
        self._axis.set_xticklabels([])
        self._axis.set_yticks(np.arange(0.5, ncols, step=1))
        self._axis.set_yticklabels([])
        self._axis.grid(True)
        self._axis.plot(*self._exit_cell, "gs", markersize=30)
        self._axis.text(*self._exit_cell, "Exit", ha="center", va="center", color="red")
        self._axis.imshow(self._maze, cmap="binary")
        self._axis.get_figure().canvas.draw()

    def render(self, block_execution=False):
        if not self._axis:
            fig, self._axis = plt.subplots(1, 1, tight_layout=True)
            fig.canvas.set_window_title("Maze")
            self._axis.set_axis_off()
        self._render()
        plt.show(block=block_execution)