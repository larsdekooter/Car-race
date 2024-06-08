from game import Game
import pygame
from network import Network
import data


def train():
    game = Game()
    network = Network()
    percentageIndex = 0
    percentages = [90, 95, 97, 99]

    while True:
        final_move = [0, 0, 0, 0, 0]
        state_old = network.get_state(game)
        move = network.getMove(state_old)
        final_move[move] = 1
        reward, done, score = game.step(final_move)
        stateNew = network.get_state(game)
        network.remember(state_old, move, reward, stateNew, done)
        if done:
            game.reset()
            game.ngames += 1
            network.ngames += 1
            network.trainLong()
            net = network.net
            rand = network.rand
            network.net = 0
            network.rand = 0

            if score > game.record:
                game.record = score

            try:
                game.percentage = round((net / (net + rand)) * 100.0, 2)
            except:
                pass
            if (
                game.percentage > percentages[percentageIndex]
                and percentageIndex < len(percentages) - 1
            ):
                percentageIndex += 1
                network.model.save()

            print(
                "Game",
                game.ngames,
                "Score",
                score,
                "Record",
                game.record,
                "%",
                game.percentage,
                "steps",
                network.decayStep,
            )
            if game.ngames % data.targetUpdate == 0:
                network.trainer.trainTarget()


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


train()
