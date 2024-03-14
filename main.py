from game import Game
from network import Network

game = Game()
network = Network()

record = 0
while True:
    oldState = network.getState(game)
    finalmove = network.getMove(oldState)
    reward, done, score = game.step(finalmove)
    stateNew = network.getState(game)
    network.trainshort(oldState, finalmove, reward, stateNew, done)
    network.remember(oldState, finalmove, reward, stateNew, done)

    if done:
        net = network.net
        rand = network.rand
        network.net = 0
        network.rand = 0
        game.car.reset()
        network.ngames += 1
        network.trainLong()
        if score > record:
            record = score

        print(
            "Game",
            network.ngames,
            "Score",
            score,
            "REcord",
            record,
            "%",
            round(net / (net + rand) * 100.0, 2),
        )
