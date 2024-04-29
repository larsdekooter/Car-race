from game import Game
import pygame
from network import Network
import sys


def train():
    game = Game()
    network = Network()
    record = 0
    while True:
        state_old = network.get_state(game)
        final_move = network.getMove(state_old)
        reward, done, score = game.step(final_move)
        stateNew = network.get_state(game)
        print(translate_moves(final_move), reward, game.car.speed)
        network.trainShort(state_old, final_move, reward, stateNew, done)
        network.remember(state_old, final_move, reward, stateNew, done)
        if done:
            game.reset()
            network.ngames += 1
            network.trainLong()
            net = network.net
            rand = network.rand
            network.net = 0
            network.rand = 0

            if score > record:
                record = score

            print(
                "Game",
                network.ngames,
                "Score",
                score,
                "Record",
                record,
                "%",
                ((net / (net + rand)) * 100.0).__round__(2),
            )


def get_moves():
    keys = pygame.key.get_pressed()
    final_move = [0, 0, 0, 0]
    if keys[pygame.K_LEFT]:
        final_move[2] = 1
    elif keys[pygame.K_RIGHT]:
        final_move[3] = 1
    elif keys[pygame.K_UP]:
        final_move[1] = 1
    elif keys[pygame.K_DOWN]:
        final_move[0] = 1
    return final_move

def translate_moves(move):
    if move[0] == 1:
        return "Backwards"
    elif move[1] == 1:
        return "Forwards"
    if move[2] == 1:
        return "Left"
    if move[3] == 1:
        return "Right"
train()
