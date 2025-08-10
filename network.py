import torch.nn as nn
import torch
import numpy as np
import data
import torch.optim as optim
from collections import deque
from game import Game
from util import getDistanceToPoint, getDistanceToLine, getAngleToLine
import random


class DQN(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(inp, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, out),
        )

    def forward(self, x):
        return self.layers(x)


class Network:
    def __init__(self):
        self.model = DQN(14, 4)
        self.targetModel = DQN(14, 4)
        self.updateTargetModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=data.lr)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.step = 0
        self.randomPerGame = [0]
        self.aiPerGame = [0]
        self.ngames = 0

    def getState(self, game: Game):
        distanceToWalls = []
        for raycast in game.car.raycasts:
            intersections = raycast.point(game.circuit)
            distances = []
            for point in intersections:
                distances.append(
                    getDistanceToPoint(game.car.x, game.car.y, point[0], point[1])
                )
            distanceToWalls.append(distances)

        currentLine = game.pointLines[game.car.currentLine]
        distanceToLine = getDistanceToLine(game.car.x, game.car.y, currentLine)
        angleToLine = getAngleToLine(
            game.car.x, game.car.y, game.car.angle, currentLine
        )

        state = [
            np.min(distanceToWalls[0]) if len(distanceToWalls[0]) > 0 else 1000,
            np.min(distanceToWalls[1]) if len(distanceToWalls[1]) > 0 else 1000,
            np.min(distanceToWalls[2]) if len(distanceToWalls[2]) > 0 else 1000,
            np.min(distanceToWalls[3]) if len(distanceToWalls[3]) > 0 else 1000,
            np.min(distanceToWalls[4]) if len(distanceToWalls[4]) > 0 else 1000,
            np.min(distanceToWalls[5]) if len(distanceToWalls[5]) > 0 else 1000,
            np.min(distanceToWalls[6]) if len(distanceToWalls[6]) > 0 else 1000,
            np.min(distanceToWalls[7]) if len(distanceToWalls[7]) > 0 else 1000,
            distanceToLine,
            angleToLine,
            game.car.x,
            game.car.y,
            game.car.speed,
            game.car.angle,
        ]

        return np.array(state, dtype=np.float32)

    def getMove(self, state):
        self.epsilon = data.minEpsilon + (data.maxEpsilon - data.minEpsilon) * np.exp(
            -data.decayRate * self.step
        )
        self.step += 1
        if np.random.rand() < self.epsilon:
            self.randomPerGame[self.ngames] += 1
            return random.randint(0, 3)
        else:
            self.aiPerGame[self.ngames] += 1
            return self.model(torch.tensor(state, dtype=torch.float32)).argmax().item()

    def train(self, state, newState, action, reward, done):
        self.memory.append((state, newState, action, reward, done))
        if len(self.memory) < data.batchSize:
            return

        batch = random.sample(self.memory, data.batchSize)
        states, newStates, actions, rewards, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        newStates = torch.tensor(np.array(newStates), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        currentQValues = self.model(states).gather(1, actions)
        nextQValues = self.targetModel(newStates).max(1)[0].detach()
        targetQValues = rewards + data.gamma * nextQValues * (1.0 - dones)

        loss = self.criterion(currentQValues, targetQValues.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        if self.step % 100 == 0:
            self.updateTargetModel()

    def updateTargetModel(self):
        self.targetModel.load_state_dict(self.model.state_dict())
