import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import deque
from game import Game
import numpy as np
import util
import random
import data
from copy import deepcopy


class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, output_size):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size),
        )

    def forward(self, x):
        return self.stack(x)

    def save(self, filename="model.pth"):
        modelFolderPath = "./model"
        if not os.path.exists(modelFolderPath):
            os.makedirs(modelFolderPath)

        filename = os.path.join(modelFolderPath, filename)
        torch.save(self.state_dict(), filename)


class QTrainer:
    def __init__(self, model: LinearQNet, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.targetModel = deepcopy(model)

    def trainTarget(self):
        self.targetModel.load_state_dict(self.model.state_dict())

    def trainStep(self, states, actions, rewards, nextStates, dones):
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        nextStates = torch.tensor(np.array(nextStates), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        currentQValues = self.model(states).gather(1, actions)
        nextStateActions = self.model(nextStates).max(1)[1].unsqueeze(1).to(torch.int64)
        nextQValues = self.targetModel(nextStates).gather(1, nextStateActions).detach()
        expectedQValues = rewards + (self.gamma * nextQValues * (1 - dones))

        loss = self.criterion(currentQValues, expectedQValues)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Network:
    def __init__(self, load=False):
        self.gamma = data.gamma
        self.memory = deque(maxlen=100_000)
        self.model = LinearQNet(11, data.hiddenSize, data.hiddenSize, 4)
        self.trainer = QTrainer(self.model, lr=data.lr, gamma=self.gamma)
        self.maxEpsilon = data.maxEpsilon
        self.minEpsilon = data.minEpsilon
        self.decayRate = data.decayRate
        self.net = 0
        self.rand = 0
        self.decayStep = 0
        if load:
            self.model.load_state_dict(torch.load("./model/model.pth"))

    def get_state(self, game: Game):
        distanceToWalls = []
        for raycast in game.car.raycastlines:
            points = raycast.get_collision_points(game.circuitLines)
            distances = []
            for point in points:
                distances.append(util.getDistanceToPoint(game.car.x, game.car.y, point))
            distanceToWalls.append(distances)

        currentLine = game.pointLines[game.car.currentLine]
        distance = util.getShortestDistanceToLine(game.car.x, game.car.y, currentLine)
        game.car.lastDistance = distance

        state = [
            np.min(distanceToWalls[0]) if len(distanceToWalls[0]) > 0 else 1000,
            np.min(distanceToWalls[1]) if len(distanceToWalls[1]) > 0 else 1000,
            np.min(distanceToWalls[2]) if len(distanceToWalls[2]) > 0 else 1000,
            np.min(distanceToWalls[3]) if len(distanceToWalls[3]) > 0 else 1000,
            np.min(distanceToWalls[4]) if len(distanceToWalls[4]) > 0 else 1000,
            np.min(distanceToWalls[5]) if len(distanceToWalls[5]) > 0 else 1000,
            np.min(distanceToWalls[6]) if len(distanceToWalls[6]) > 0 else 1000,
            np.min(distanceToWalls[7]) if len(distanceToWalls[7]) > 0 else 1000,
            distance,
            # game.car.x,
            # game.car.y,
            game.car.speed,
            game.car.angle,
        ]

        return np.array(state, dtype=float)

    def getMove(self, state):
        epsilon = self.minEpsilon + (self.maxEpsilon - self.minEpsilon) * np.exp(
            -self.decayRate * self.decayStep
        )

        if np.random.rand() < epsilon:
            move = random.randint(0, 3)
            # final_move[choice] = 1
            self.rand += 1
        else:
            with torch.no_grad():
                state0 = torch.tensor(state, dtype=torch.float)
                prediction = self.model(state0)
                noise = torch.randn(prediction.size()) * 0.1
                prediction += noise
                move = prediction.argmax().item()
                self.net += 1
        # state0 = torch.tensor(state, dtype=torch.float)
        # prediciton = self.model(state0)
        # move = torch.argmax(prediciton).item()
        # final_move[move] = 1
        # self.net += 1
        self.decayStep += 1
        return move

    def getMoveSelf(self, state):
        self.model.train()
        final_move = [0, 0, 0, 0]
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        return move

    def trainShort(self, state, action, reward, next_state, done):
        self.trainer.trainStep(state, action, reward, next_state, done)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def trainLong(self):
        if len(self.memory) < data.batchSize:
            return

        batch = random.sample(self.memory, data.batchSize)

        states, actions, rewards, next_states, dones = zip(*batch)
        self.trainer.trainStep(states, actions, rewards, next_states, dones)
