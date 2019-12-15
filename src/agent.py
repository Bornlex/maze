#coding: utf-8

import random
import numpy as np
from collections import deque
from tensorflow import keras


class Agent(object):
    def __init__(self, state_space):
        self._batch_size   = 32
        self._action_space = 4
        self._state_space  = state_space
        self._epsilon      = 1
        self._eps_decay    = 0.95
        self._eps_minimum  = 0.01
        self._gamma        = 0.9
        self._memory       = deque(maxlen=10000)
        self._model        = self.build_model()

    def build_model(self):
        input_x = keras.layers.Input(shape=(self._state_space,))
        x       = keras.layers.Dense(32, activation="relu")(input_x)
        x       = keras.layers.Dense(32, activation="relu")(x)
        x       = keras.layers.Dense(self._action_space, activation="linear")(x)
        model   = keras.models.Model(inputs=[input_x], outputs=[x])
        model.compile(optimizer="adam", loss="mse")
        return model

    def remember(self, state, action, reward, next_state, done):
        self._memory.append((
            state,
            action,
            reward,
            next_state,
            done
        ))

    def act(self, state):
        if np.random.rand() <= self._epsilon:
            return random.randrange(self._action_space)
        actions = self._model.predict(state)
        return np.argmax(actions)

    def replay(self):
        if len(self._memory) < self._batch_size:
            return
        batch    = random.sample(self._memory, self._batch_size)
        states   = np.array([b[0] for b in batch])
        actions  = np.array([b[1] for b in batch])
        rewards  = np.array([b[2] for b in batch])
        n_states = np.array([b[3] for b in batch])
        dones    = np.array([b[4] for b in batch])
        targets  = rewards + self._gamma * (np.amax(self._model.predict(n_states))) * (1 - dones)
        targets_full = self._model.predict(states)
        indices = np.array([i for i in range(self._batch_size)])
        targets_full[[indices], [actions]] = targets
        self._model.fit(states, targets_full, epochs=1, verbose=0)
        if self._epsilon > self._eps_minimum:
            self._epsilon *= self._eps_decay