from game import Game
from network import Network
import matplotlib.pyplot as plt
from IPython import display
import numpy as np


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    network = Network()
    game = Game()
    game.start()
    game.step([0, 0, 0, 0])

    while True:
        state_old = network.get_state(game)
        final_move = network.translate_moves(network.get_action(state_old))
        reward, done, score = game.step(final_move)
        state_new = network.get_state(game)
        network.train_short_memory(state_old, final_move, reward, state_new, done)
        network.remember(state_old, final_move, reward, state_new, done)

        if done:
            moves = game.car.moves
            game.reset()
            random_this_game = network.random_this_game
            network_this_game = network.network_this_game
            network.random_this_game = 0
            network.network_this_game = 0
            network.n_games += 1
            network.train_long_memory()

            if score > record:
                record = score
                network.model.save()

            most_occuring_move = max(moves, key=moves.count)

            print(
                "Game",
                network.n_games,
                "Score",
                score,
                "Record",
                record,
                "Most occuring move",
                most_occuring_move,
                "Contains other moves",
                len(list(filter(lambda move: move != most_occuring_move, moves))) > 0,
                "random",
                network.minEpsilon
                + (network.maxEpsilon - network.minEpsilon)
                * np.exp(-network.decayRate * network.n_games),
                "network %",
                round(
                    (network_this_game / (network_this_game + random_this_game) * 100),
                    2,
                ),
            )

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / (network.n_games)
            plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)
            if network.n_games % 100 == 0:
                network.model.save()


plt.ion()


def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Number of games")
    plt.ylabel("Score")
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(0.1)


def run():
    network = Network()
    game = Game()
    game.start()
    game.step([0, 0, 0, 0])

    while True:
        state_old = network.get_state(game)
        final_move = network.translate_moves(network.get_action(state_old))
        reward, done, score = game.step(final_move)
        state_new = network.get_state(game)


train()
# run()
