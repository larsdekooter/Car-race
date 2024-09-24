from game import Game
from network import Network
from tqdm import tqdm
import data

game = Game(False)
network = Network()

state = network.getState(game)

for i in tqdm(range(436675)):
    action = network.getMove(state)
    finalmove = [0, 0, 0, 0]
    finalmove[action] = 1
    reward, done, score = game.step(finalmove)
    newState = network.getState(game)
    network.train(state, newState, action, reward, done)
    state = newState
    if done:
        game.reset()
        state = network.getState(game)
        network.ngames += 1
        network.aiPerGame.append(0)
        network.randomPerGame.append(0)
        if network.epsilon <= data.minEpsilon:
            break

game = Game(True)
state = network.getState(game)
moves = []
while True:
    action = network.getMove(state)
    finalmove = [0, 0, 0, 0]
    finalmove[action] = 1
    reward, done, score = game.step(finalmove)
    newState = network.getState(game)
    network.train(state, newState, action, reward, done)
    state = newState
    moves.append(action)
    if done:
        print(
            f"Games: {network.ngames}, Score: {score}, Percentage: {round(100.0 * network.aiPerGame[network.ngames] / (network.aiPerGame[network.ngames] + network.randomPerGame[network.ngames]), 2)}%, Epsilon: {network.epsilon}"
        )
        game.reset()
        state = network.getState(game)
        network.ngames += 1
        network.aiPerGame.append(0)
        network.randomPerGame.append(0)
        file = open("moves.txt", "w")
        file.write(str(moves))
        file.close()
        break
