import torch
import torch.nn as nn
from collections import deque
from game import Game
import numpy as np
import random
from time import time
import math


class Net(nn.Module):
    def __init__(self, inputs, hiddens, outputs):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(inputs, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, outputs),
        )

    def forward(self, x):
        return self.stack(x)


class Trainer:
    def __init__(self, lr, gamma, model):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def trainStep(self, state, action, reward, nextState, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        nextState = torch.tensor(np.array(nextState), dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            nextState = torch.unsqueeze(nextState, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(nextState[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


class Network:
    def __init__(self):
        self.ngames = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=100_000)
        self.model = Net(25, 256, 4)
        self.trainer = Trainer(1e-4, self.gamma, self.model)
        self.maxE = 1
        self.minE = 0.1
        self.decayR = 1e-5
        self.decayS = 0
        self.net = 0
        self.rand = 0

    def getState(self, game: Game):
        state: list[tuple[float]] = []
        for line in game.car.raycastlines:
            for point in line.get_collision_points(game.circuit):
                state.append(point)
        state = list(
            map(
                lambda point: math.sqrt(
                    (game.car.x - point[0]) ** 2 + (game.car.y - point[1]) ** 2
                ),
                state,
            )
        )
        state.extend([game.car.x, game.car.y, game.car.speed, game.car.angle])
        state = state + ([0] * (25 - len(state)))
        return np.array(state, dtype=float)

    def getMove(self, state):
        epsilon = self.minE + (self.maxE - self.minE) * np.exp(
            -self.decayR * self.decayS
        )
        finalmove = [0, 0, 0, 0]

        if np.random.rand() < epsilon:
            choice = random.randint(0, 3)
            finalmove[choice] = 1
            self.rand += 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            finalmove[move] = 1
            self.net += 1
        self.decayS += 1
        return finalmove

    def trainshort(self, state, action, reward, nextState, done):
        self.trainer.trainStep(state, action, reward, nextState, done)

    def remember(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))

    def trainLong(self):
        if len(self.memory) > 10_000:
            sample = random.sample(self.memory, 10_000)
        else:
            sample = self.memory

        states, actions, rewards, nextStates, dones = zip(*sample)
        self.trainer.trainStep(states, actions, rewards, nextStates, dones)
