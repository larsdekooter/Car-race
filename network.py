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
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def trainStep(self, state, action, reward, nextState, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(nextState), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(
                    self.model(next_state[idx])
                )

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()


class Network:
    def __init__(self):
        self.gamma = data.gamma
        self.memory = deque(maxlen=100_000)
        self.model = LinearQNet(13, data.hiddenSize, data.hiddenSize, 4)
        self.trainer = QTrainer(self.model, lr=data.lr, gamma=self.gamma)
        self.maxEpsilon = data.maxEpsilon
        self.minEpsilon = data.minEpsilon
        self.decayRate = data.decayRate
        self.decayStep = 0
        self.net = 0
        self.rand = 0
        self.decayStep = 0

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
            game.car.x,
            game.car.y,
            game.car.speed,
            game.car.angle,
        ]

        return np.array(state, dtype=float)

    def getMove(self, state):
        epsilon = self.minEpsilon + (self.maxEpsilon - self.minEpsilon) * np.exp(
            -self.decayRate * self.decayStep
        )

        final_move = [0, 0, 0, 0]
        if np.random.rand() < epsilon:
            choice = random.randint(0, 3)
            final_move[choice] = 1
            self.rand += 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            self.net += 1
        # state0 = torch.tensor(state, dtype=torch.float)
        # prediciton = self.model(state0)
        # move = torch.argmax(prediciton).item()
        # final_move[move] = 1
        # self.net += 1
        self.decayStep += 1
        return final_move

    def trainShort(self, state, action, reward, next_state, done):
        self.trainer.trainStep(state, action, reward, next_state, done)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def trainLong(self):
        if len(self.memory) > data.batchSize:
            mini_sample = random.sample(self.memory, data.batchSize)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.trainStep(states, actions, rewards, next_states, dones)
