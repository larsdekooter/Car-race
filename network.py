from game import Game
import numpy as np
from collections import deque
import random
import math
from model import Linear_Qnet, QTrainer
import torch

moves = ["UP", "DOWN", "LEFT", "RIGHT"]

DISTANCE = 40


class Network:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=100_000)
        self.model = Linear_Qnet(10, 256, 4)
        self.model.train(False)
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)

        pass

    def get_state(self, game: Game):
        x, y = game.car.x, game.car.y
        intersections = game.get_intersections()
        distances = []
        for line in intersections:
            line_distances = []
            for inter in line:
                line_distances.append(self.get_distance_to_intersection(x, y, inter))
            distances.append(line_distances)
        state = [
            # Wall dangers
            np.min(distances[0]) < DISTANCE if len(distances[0]) > 0 else False,
            np.min(distances[1]) < DISTANCE if len(distances[1]) > 0 else False,
            np.min(distances[2]) < DISTANCE if len(distances[2]) > 0 else False,
            np.min(distances[3]) < DISTANCE if len(distances[3]) > 0 else False,
            np.min(distances[4]) < DISTANCE if len(distances[4]) > 0 else False,
            np.min(distances[5]) < DISTANCE if len(distances[5]) > 0 else False,
            np.min(distances[6]) < DISTANCE if len(distances[6]) > 0 else False,
            np.min(distances[7]) < DISTANCE if len(distances[7]) > 0 else False,
            game.car.speed,
            game.car.angle,
        ]
        return list(np.array(state, dtype=int))

    def get_action(self, state):
        self.epsilon = 40 - self.n_games
        final_move = [0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            # print(torch.argmin(prediction).item(), torch.argmax(prediction).item())
            move = torch.argmax(prediction).item()
            print(move)
            final_move[move] = 1
        return final_move

    def translate_moves(self, moves: list):
        mvs = ["UP", "DOWN", "LEFT", "RIGHT"]
        retlist = []
        for move in moves:
            i = moves.index(move)
            retlist.append(mvs[i])
        return moves

    def get_distance_to_intersection(self, x, y, point):
        pointX, pointY = point[0], point[1]
        dx = pointX - x
        dy = pointY - y
        return math.sqrt(dx * dx + dy * dy)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > 1000:
            mini_sample = random.sample(self.memory, 1000)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
