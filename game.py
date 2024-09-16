from car import Car
import pygame
import circuit
import time
import data


class Game:
    def __init__(self):
        self.s = time.time()
        self.car = Car()
        self.starttime = time.time()
        self.closestDistance = None
        self.record = 0
        self.ngames = 0
        self.percentage = 0
        self.pointLines = circuit.point_lines()
        self.circuitLines = circuit.circuit()
        self.screen = None

    def infoLines(self):
        pygame.draw.line(self.screen, "white", (1200, 0), (1200, 720), 1)
        self.text(
            "Speed: "
            + str(
                round(self.car.speed, 4)
                if self.car.speed >= 0
                else -round(self.car.speed, 4)
            ),
            1210,
            5,
        )
        self.text("Angle: " + str(self.car.angle), 1210, 35)
        self.text(
            "dX: "
            + str(
                int(self.car.currentDistance(self.pointLines[self.car.currentLine])[0])
            ),
            1210,
            70,
        )
        self.text("rew: " + str(int(self.car.rewardThisGame)), 1210, 105)
        self.text("x " + str(int(self.car.x)), 1210, 140)
        self.text("y " + str(int(self.car.y)), 1210, 175)
        self.text("R " + str(self.record), 1210, 210)
        self.text("nG " + str(self.ngames), 1210, 245)
        self.text(str(self.percentage) + "%", 1210, 280)
        self.text("h" + str(round((time.time() - self.s) / 3600, 2)), 1210, 315)
        self.text(str(self.car.points), 5, 5)
        self.text(str(round(time.time() - self.starttime, 0)), 640, 360)

    def text(self, text: str, x: int, y: int):
        fo = self.font.render(text, True, "white", "black")
        foT = fo.get_rect()
        foT.centerx += x
        foT.centery += y
        self.screen.blit(fo, foT)

    def render(self):
        # Initialize pygame if not already initialized
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((1380, 720))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font("arial.ttf", 32)

        self.handleEvents()
        self.screen.fill("black")

        # Draw all the lines
        for line in self.circuitLines:
            line.draw(self.screen)
        self.pointLines[self.car.currentLine].draw(self.screen)
        for raycast in self.car.raycastlines:
            raycast.draw(self.screen)

        pygame.draw.rect(self.screen, "orange", self.car.hitbox, 1)

        rotatedCarImg = pygame.transform.rotate(self.car.img, self.car.angle)
        self.screen.blit(rotatedCarImg, (self.car.x, self.car.y))

        self.infoLines()

        pygame.display.flip()
        self.clock.tick(60)

    def step(self, moves, render=False):
        self.car.move(moves)
        reward = 0
        hitbox = pygame.Rect(self.car.x, self.car.y, 20, 20)
        done = self.checkCircuitCollisions(hitbox)
        reward = self.handleRewards(hitbox)

        if time.time() - self.starttime > data.time:
            done = True
        if render:
            self.render()
        return reward, done, self.car.points

    def handleRewards(self, hitbox):
        reward = data.timeStepPenalty
        if self.checkPointCollissions(hitbox):
            distanceDriven = self.car.getDistanceDriven()
            if distanceDriven > 1:
                reward = data.lineReward / (distanceDriven * data.distancePenalty)
            else:
                reward = 100
        if self.checkCircuitCollisions(hitbox):
            reward = data.hitCost

        return reward

    def handleEvents(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    def checkCircuitCollisions(self, hitbox: pygame.Rect):
        for line in self.circuitLines:
            if bool(hitbox.clipline(line.start, line.end)):
                return True
        return False

    def checkPointCollissions(self, hitbox: pygame.Rect):
        for line in self.pointLines:
            if not line.isDrawn:
                pass
            elif hitbox.clipline(line.start, line.end):
                index = line.i
                if self.car.lastLine == index:
                    pass
                else:
                    self.car.lastLine = index
                    self.car.points += 1
                    self.car.currentLine += 1
                    if self.car.currentLine == len(self.pointLines):
                        self.car.currentLine = 0
                    return True

    def reset(self):
        self.car.reset()
        self.starttime = time.time()
        self.closestDistance = None
