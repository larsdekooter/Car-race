from circuit import circuit, pointLines
import pygame
import data
import math
from time import time
from util import findIntersections
import numpy as np


class RaycastLine:
    def __init__(self, x, y, direction, width=0):
        self.x = x
        self.y = y
        length = data.raycastLength
        self.end = (x + direction[0] * length, y + direction[1] * length)
        self.width = width
        self.start = (x, y)

    def draw(self, screen):
        pygame.draw.line(screen, "white", self.start, self.end, self.width)

    def point(self, lines):
        return findIntersections(self, lines)


class Car:
    def __init__(self) -> None:
        self.reset()
        self.img = pygame.transform.scale(pygame.image.load("car.png"), (25, 25))

    def reset(self):
        self.position = (data.x, data.y)
        self.x = data.x
        self.y = data.y
        self.angle = 0
        self.speed = 0
        self.acceleration = 0.2
        self.deceleration = 0.5
        self.turnSpeed = 1
        self.maxSpeed = 10
        self.score = 0
        self.currentLine = 0
        self.initRaycasts(1)

        self.updateHitbox()

    def move(self, moves):
        if moves[0] == 1:  # backwards
            self.speed += self.acceleration
            if self.speed > self.maxSpeed:
                self.speed = self.maxSpeed
        elif moves[1] == 1:  # forwards
            self.speed -= self.deceleration
            if self.speed < -self.maxSpeed:
                self.speed = -self.maxSpeed
        else:
            if self.speed > 0:
                self.speed -= self.deceleration
            elif self.speed < 0:
                self.speed += self.deceleration

        if moves[2] == 1:  # left
            self.angle += self.turnSpeed
            if self.angle > 360:
                self.angle -= 360
        elif moves[3] == 1:  # right
            self.angle -= self.turnSpeed
            if self.angle < -360:
                self.angle += 360

        radians = self.angle * (math.pi / 180)
        self.x += self.speed * math.sin(radians)
        self.y += self.speed * math.cos(radians)
        self.updateHitbox()

    def updateHitbox(self):
        self.hitbox = pygame.Rect(self.x, self.y, 25, 25)

    def initRaycasts(self, width=0):
        self.raycasts = [
            RaycastLine(
                self.x + 10,
                self.y + 10,
                self.getLookingDirection(self.angle),
                width,
            ),
            RaycastLine(
                self.x + 10,
                self.y + 10,
                (
                    -self.getLookingDirection(self.angle)[0],
                    -self.getLookingDirection(self.angle)[1],
                ),
                width,
            ),
            RaycastLine(
                self.x + 10,
                self.y + 10,
                self.getLookingDirection(self.angle + 90),
                width,
            ),
            RaycastLine(
                self.x + 10,
                self.y + 10,
                self.getLookingDirection(self.angle - 90),
                width,
            ),
            RaycastLine(
                self.x + 10,
                self.y + 10,
                self.getLookingDirection(self.angle - 45),
                width,
            ),
            RaycastLine(
                self.x + 10,
                self.y + 10,
                self.getLookingDirection(self.angle + 45),
                width,
            ),
            RaycastLine(
                self.x + 10,
                self.y + 10,
                (
                    -self.getLookingDirection(self.angle - 45)[0],
                    -self.getLookingDirection(self.angle - 45)[1],
                ),
                width,
            ),
            RaycastLine(
                self.x + 10,
                self.y + 10,
                (
                    -self.getLookingDirection(self.angle + 45)[0],
                    -self.getLookingDirection(self.angle + 45)[1],
                ),
                width,
            ),
        ]

    def getLookingDirection(self, angle):
        radians = math.radians(angle)
        return (math.sin(radians), math.cos(radians))


class Game:
    def __init__(self, render: bool):
        self.shouldRender = render
        self.circuit = circuit()
        self.pointLines = pointLines()
        self.screen = None
        self.car = Car()
        self.reset()

    def reset(self):
        self.car.reset()
        self.startTime = time()

    def text(self, text, x, y):
        self.screen.blit(self.font.render(text, True, "white"), (x, y))

    def infoLines(self):
        self.text(f"{self.car.score}", 5, 5)
        self.text(f"{int(time() - self.startTime)}", 640, 360)

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((1280, 720))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font("arial.ttf", 32)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill("black")

        for line in self.circuit:
            line.draw(self.screen)

        img = pygame.transform.rotate(self.car.img, self.car.angle)
        self.screen.blit(img, (self.car.x, self.car.y))

        # self.pointLines[self.car.currentLine].draw(self.screen)
        for line in self.pointLines:
            line.draw(self.screen)

        self.infoLines()

        pygame.display.flip()
        self.clock.tick(60)

    def step(self, moves):
        self.car.move(moves)
        self.car.initRaycasts(1)

        wall, point, reversePoint = self.checkCollision()

        # checkpoint handling
        if point:
            self.car.currentLine += 1
            self.car.score += 1

        if self.car.currentLine >= len(self.pointLines):
            self.car.currentLine = 0

        # compute reward and done
        reward, done = self.reward(wall, point, reversePoint)

        if self.shouldRender:
            self.render()

        if done:
            # keep a deterministic reset behaviour (caller expects to handle logging)
            return reward, True, self.car.score

        # normal step return
        return reward, False, self.car.score

    def reward(self, wall, point, reversePoint):
        if wall:
            return -1000.0, True, self.car.score
        reward = 0.0
        if point:
            reward += 500.0

        # Small negative reward for each step
        reward -= 0.05

        # Reward for moving closer to the next checkpoint
        currentLine = self.pointLines[self.car.currentLine]
        prev_dist = getattr(self.car, "prev_dist", None)
        dist = getDistanceToLine(self.car.x, self.car.y, currentLine)
        if prev_dist is not None:
            reward += (prev_dist - dist) * 2  # reward for reducing distance
        self.car.prev_dist = dist

        if self.car.currentLine == 0 and point:
            reward += 2000.0

        if time() - self.startTime > data.timeLimit:
            return reward, True, self.car.score

        return reward, False, self.car.score

    def checkCollision(self):
        for line in self.circuit:
            if bool(self.car.hitbox.clipline(line.start, line.end)):
                return True, False, False
        currentLine = self.pointLines[self.car.currentLine]
        if bool(self.car.hitbox.clipline(currentLine.start, currentLine.end)):
            return False, True, False
        for i in range(len(self.pointLines.tolist())):
            if i < self.car.currentLine and bool(self.car.hitbox.clipline(self.pointLines[i].start, self.pointLines[i].end)):
                return False, False, True
        return False, False, False
