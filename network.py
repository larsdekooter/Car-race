import torch
import torch.nn as nn
import numpy as np
import data
import torch.optim as optim
from collections import deque
from game import Game
import util
import random


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(data.inputSize, data.hiddenSize),
            nn.ReLU(),
            nn.Linear(data.hiddenSize, data.hiddenSize),
            nn.ReLU(),
            nn.Linear(data.hiddenSize, data.outputSize),
        )

    def forward(self, x):
        return self.stack(x)


class Network:
    def __init__(self):
        self.model = DQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=data.lr)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=100_000)
        self.step = 0
        self.randomPerGame = [0]
        self.aiPerGame = [0]
        self.ngames = 0
        self.targetModel = DQN()
        self.targetModel.load_state_dict(self.model.state_dict())

    def getState(self, game: Game):
        distanceToWalls = []
        for raycast in game.car.raycastlines:
            points = raycast.get_collision_points(game.circuitLines)
            distances = []
            for point in points:
                distances.append(util.getDistanceToPoint(game.car.x, game.car.y, point))
            distanceToWalls.append(distances)

        currentLine = game.pointLines[game.car.currentLine]
        distance, xDistance, yDistance = util.getShortestDistanceToLine(
            game.car.x, game.car.y, currentLine
        )
        game.car.lastDistance = distance

        state = [
            np.min(distanceToWalls[0] if len(distanceToWalls[0]) > 0 else 1000),
            np.min(distanceToWalls[1] if len(distanceToWalls[1]) > 0 else 1000),
            np.min(distanceToWalls[2] if len(distanceToWalls[2]) > 0 else 1000),
            np.min(distanceToWalls[3] if len(distanceToWalls[3]) > 0 else 1000),
            np.min(distanceToWalls[4] if len(distanceToWalls[4]) > 0 else 1000),
            np.min(distanceToWalls[5] if len(distanceToWalls[5]) > 0 else 1000),
            np.min(distanceToWalls[6] if len(distanceToWalls[6]) > 0 else 1000),
            np.min(distanceToWalls[7] if len(distanceToWalls[7]) > 0 else 1000),
            distance,
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
        dones = torch.tensor(dones, dtype=torch.bool)

        currentQValues = self.model(states).gather(1, actions)
        nextQValues = self.targetModel(newStates).max(1)[0]
        targetQValues = rewards + data.gamma * nextQValues * (~dones)

        loss = self.criterion(currentQValues, targetQValues.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.step % 100 == 0:
            self.targetModel.load_state_dict(self.model.state_dict())
