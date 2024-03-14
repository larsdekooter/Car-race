import math
import pygame
from raycast import Raycast


class Car:
    def __init__(self):
        self.reset()
        self.img = pygame.transform.scale(pygame.image.load("car.png"), (25, 25))
        self.maxSpeed = 7
        self.turnSpeed = 10
        self.acceleration = 0.5
        self.decellaration = -0.5

    def reset(self):
        self.x = 1050
        self.y = 550
        self.speed = 0
        self.angle = 0
        self.updateHitbox()
        self.currentLine = 0
        self.score = 0
        self.raycastlines = []

    def updateHitbox(self):
        self.hitbox = (self.x, self.y, 20, 20)

    def move(self, move: list[int]):
        self.moveStraight(move)
        xChange, yChange = self.moveTurns(move)
        self.x += xChange
        self.y += yChange
        self.updateHitbox()
        return pygame.transform.rotate(self.img, self.angle)

    def moveTurns(self, move: list[int]):
        if move[2] == 1:
            self.angle += self.turnSpeed
            if self.angle >= 360:
                self.angle -= 360
        elif move[3] == 1:
            self.angle -= self.turnSpeed
            if self.angle <= -360:
                self.angle += 360

        radians = self.angle * (math.pi / 180)
        xChange = self.speed * math.sin(radians)
        yChange = self.speed * math.cos(radians)
        return xChange, yChange

    def moveStraight(self, move: list[int]):
        if move[1] == 1:  # forward
            self.speed += self.acceleration
            if self.speed > self.maxSpeed:
                self.speed = self.maxSpeed
        elif move[0] == 1:  # backward
            self.speed += self.decellaration
            if self.speed < -self.maxSpeed:
                self.speed = -self.maxSpeed
        else:
            if self.speed > 0:  # slow forward movement down
                self.speed += self.decellaration
            elif self.speed < 0:  # slow backward movement down
                self.speed += self.acceleration
            if abs(self.speed) < self.decellaration:  # stop the car from moving
                self.speed = 0

    def getDir(
        self,
        angleAddition=0,
    ):
        angleRadians = math.radians(self.angle + angleAddition)
        return (math.sin(angleRadians), math.cos(angleRadians))

    def drawRaycasts(self, screen):
        self.raycastlines = [
            Raycast(self.x + 10, self.y + 10, screen, self.getDir(0)),
            Raycast(self.x + 10, self.y + 10, screen, self.getDir(45)),
            Raycast(self.x + 10, self.y + 10, screen, self.getDir(90)),
            Raycast(self.x + 10, self.y + 10, screen, self.getDir(135)),
            Raycast(self.x + 10, self.y + 10, screen, self.getDir(180)),
            Raycast(self.x + 10, self.y + 10, screen, self.getDir(225)),
            Raycast(self.x + 10, self.y + 10, screen, self.getDir(270)),
            Raycast(self.x + 10, self.y + 10, screen, self.getDir(315)),
        ]
