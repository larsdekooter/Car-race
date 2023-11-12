from game import Game
import pygame


def train():
    game = Game()
    while True:
        done = game.step(get_moves())
        if done:
            game.reset()


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


train()
