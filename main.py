from game import Game
import pygame
from network import Network
import data
import matplotlib.pyplot as plt


game = Game()
network = Network()
state_old = network.getState(game)
while True:
    final_move = [0, 0, 0, 0, 0]
    move = network.getMove(state_old)
    final_move[move] = 1
    reward, done, score = game.step(final_move)
    stateNew = network.getState(game)
    network.train(state_old, stateNew, move, reward, done)
    state_old = stateNew
    if done:
        game.reset()
        game.ngames += 1
        network.ngames += 1
        network.aiPerGame.append(0)
        network.randomPerGame.append(0)
        print(
            f"Games: {network.ngames}, Score: {score}, Percentage: {round(network.aiPerGame[network.ngames] / (network.aiPerGame[network.ngames] + network.randomPerGame[network.ngames]) * 100.0, 2)}, Epsilon: {round(network.epsilon, 3)}"
        )


def get_moves():
    keys = pygame.key.get_pressed()
    final_move = [0, 0, 0, 0]
    if keys[pygame.K_LEFT]:
        return 2
    elif keys[pygame.K_RIGHT]:
        return 3
    elif keys[pygame.K_UP]:
        return 1
    elif keys[pygame.K_DOWN]:
        return 0
    else:
        return 4


def reverseTranslateMoves(move):
    if move == "B":
        return 0
    if move == "F":
        return 1
    if move == "L":
        return 2
    if move == "R":
        return 3


def translate_moves(move):
    if move[0] == 1:
        return "Backwards"
    elif move[1] == 1:
        return "Forwards"
    if move[2] == 1:
        return "Left"
    if move[3] == 1:
        return "Right"
