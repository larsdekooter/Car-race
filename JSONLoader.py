import json


class JSONLoader:
    def __init__(self) -> None:
        with open("./data.json", "r") as file:
            data = json.load(file)
        self.lr = data["lr"]
        self.gamma = data["gamma"]
        self.hiddenSize = data["hiddenSize"]
        self.maxEpsilon = data["maxEpsilon"]
        self.minEpsilon = data["minEpsilon"]
        self.decayRate = data["decayRate"]
        self.maxSpeed = data["maxSpeed"]
        self.x = data["x"]
        self.y = data["y"]
        self.accelaration = data["accelaration"]
        self.decelaration = data["decelaration"]
        self.turnSpeed = data["turnSpeed"]
        self.lineReward = data["lineReward"]
        self.timeReward = data["timeReward"]
        self.raycastLength = data["raycastLength"]
        self.distanceReward = data["distanceReward"]
        self.hitCost = data["hitCost"]
        self.time = data["time"]
        self.batchSize = data["batchSize"]
