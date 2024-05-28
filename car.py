import pygame
import math
from raycastline import RaycastLine
import data
import util
from collections import deque


class Car:
    def __init__(self):
        self.data = data
        self.reset()

    def reset(self):
        self.stateHistory = deque(maxlen=100)
        self.positions = []
        self.maxSpeed = data.maxSpeed
        self.x = data.x
        self.y = data.y
        self.speed = 0
        self.angle = 0
        self.accelaration = data.accelaration
        self.decelaration = data.decelaration
        self.turnSpeed = data.turnSpeed
        self.img = pygame.transform.scale(pygame.image.load("car.png"), (25, 25))
        self.rect = self.img.get_rect()
        self.hitbox = self.updateHitbox()
        self.points = 0
        self.currentLine = 0
        self.lastLine = -1
        self.lastDistance = 0
        self.closestDistance = None
        self.rewardThisGame = 0

    def isRepeatingStates(self):
        currentState = ((self.x, self.y), self.speed, self.angle)
        if currentState in self.stateHistory:
            return True
        self.stateHistory.append(currentState)
        return False

    def isMovingBackwards(self, line):
        if (
            self.closestDistance
            and self.currentDistance(line)[0] > self.closestDistance
        ):
            return True
        return False

    def currentDistance(self, currentLine, update=False):
        distance = util.getShortestDistanceToLine(self.x, self.y, currentLine)
        if self.closestDistance and distance[0] < self.closestDistance:
            self.closestDistance = distance
        if update:
            self.lastDistance = distance
        return distance

    def updateHitbox(self):
        self.hitbox = (self.x, self.y, 20, 20)
        pass

    def move(self, moves):
        self.handleStraight(moves)
        xChange, yChange = self.handleTurns(moves)
        self.x += xChange
        self.y += yChange
        self.updateHitbox()
        self.positions.append((self.x, self.y))
        if len(self.positions) > 100:
            self.positions.pop(0)
        return pygame.transform.rotate(self.img, self.angle)

    def calculateDisplacement(self):
        if len(self.positions) < 2:
            return 0
        startpos = self.positions[0]
        endpos = self.positions[-1]
        displacement = (
            (endpos[0] - startpos[0]) ** 2 + (endpos[1] - startpos[1]) ** 2
        ) ** 0.5
        return displacement

    def isMovingInCircles(self):
        displacement = self.calculateDisplacement()
        if displacement < data.displacementThreshold:
            return True
        return False

    def handleStraight(self, moves):
        if moves[0] == 1:  # backwards
            self.speed += self.accelaration
            if self.speed > self.maxSpeed:
                self.speed = self.maxSpeed
        elif moves[1] == 1:  # forwards
            self.speed -= self.decelaration
            if self.speed < -self.maxSpeed:
                self.speed = -self.maxSpeed
        else:
            if self.speed > 0:
                self.speed -= self.decelaration
            elif self.speed < 0:
                self.speed += self.decelaration
            if abs(self.speed) < self.decelaration:
                self.speed = 0

    def handleTurns(self, moves):
        if moves[2] == 1:  # left
            self.angle += self.turnSpeed
            if self.angle >= 360:
                self.angle -= 360
        elif moves[3] == 1:  # right
            self.angle -= self.turnSpeed
            if self.angle <= -360:
                self.angle += 360

        radians = self.angle * (math.pi / 180)
        xChange = self.speed * math.sin(radians)
        yChange = self.speed * math.cos(radians)
        return xChange, yChange

    def getLookingDirection(self, angle):
        angleRadians = math.radians(angle)
        return (math.sin(angleRadians), math.cos(angleRadians))

    def drawRaycasts(self, screen, width=0):
        self.raycastlines = [
            RaycastLine(
                self.x + 10,
                self.y + 10,
                screen,
                self.getLookingDirection(self.angle),
                width,
            ),
            RaycastLine(
                self.x + 10,
                self.y + 10,
                screen,
                (
                    -self.getLookingDirection(self.angle)[0],
                    -self.getLookingDirection(self.angle)[1],
                ),
                width,
            ),
            RaycastLine(
                self.x + 10,
                self.y + 10,
                screen,
                self.getLookingDirection(self.angle + 90),
                width,
            ),
            RaycastLine(
                self.x + 10,
                self.y + 10,
                screen,
                self.getLookingDirection(self.angle - 90),
                width,
            ),
            RaycastLine(
                self.x + 10,
                self.y + 10,
                screen,
                self.getLookingDirection(self.angle - 45),
                width,
            ),
            RaycastLine(
                self.x + 10,
                self.y + 10,
                screen,
                self.getLookingDirection(self.angle + 45),
                width,
            ),
            RaycastLine(
                self.x + 10,
                self.y + 10,
                screen,
                (
                    -self.getLookingDirection(self.angle - 45)[0],
                    -self.getLookingDirection(self.angle - 45)[1],
                ),
                width,
            ),
            RaycastLine(
                self.x + 10,
                self.y + 10,
                screen,
                (
                    -self.getLookingDirection(self.angle + 45)[0],
                    -self.getLookingDirection(self.angle + 45)[1],
                ),
                width,
            ),
        ]
