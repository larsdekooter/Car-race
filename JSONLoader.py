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
        print(
            "lr",
            self.lr,
            "gamma",
            self.gamma,
            "hidden",
            self.hiddenSize,
            "maxepsilon",
            self.maxEpsilon,
            "minepsilon",
            self.minEpsilon,
            "decayrate",
            self.decayRate,
            "maxspeed",
            self.maxSpeed,
            "x",
            self.x,
            "y",
            self.y,
            "accel",
            self.accelaration,
            "decel",
            self.decelaration,
            "turnspeed",
            self.turnSpeed,
            "line",
            self.lineReward,
            "time",
            self.timeReward,
            "length",
            self.raycastLength,
        )
