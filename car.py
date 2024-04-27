import pygame
import math
from raycastline import RaycastLine
import data


class Car:
    def __init__(self):
        self.data = data
        self.reset()

    def reset(self):
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
        self.lastDistance = None
        self.rewardThisGame = 0

    def updateHitbox(self):
        self.hitbox = (self.x, self.y, 20, 20)
        pass

    def move(self, moves):
        self.handleStraight(moves)
        xChange, yChange = self.handleTurns(moves)
        self.x += xChange
        self.y += yChange
        self.updateHitbox()
        return pygame.transform.rotate(self.img, self.angle)

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
