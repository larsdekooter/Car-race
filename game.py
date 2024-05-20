from car import Car
import pygame
import circuit
import time
import data


class Game:
    def __init__(self):
        self.s = time.time()
        self.car = Car()
        pygame.init()
        self.screen = pygame.display.set_mode((1380, 720))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font("arial.ttf", 32)
        self.circuitLines = circuit.circuit(self.screen)
        self.pointLines = circuit.point_lines(self.screen)
        self.starttime = time.time()
        self.car.drawRaycasts(self.screen)
        self.closestDistance = None
        self.record = 0
        self.ngames = 0
        self.percentage = 0

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
            + str(int(self.car.currentDistance(self.pointLines[self.car.currentLine]))),
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

    def text(self, text: str, x: int, y: int):
        fo = self.font.render(text, True, "white", "black")
        foT = fo.get_rect()
        foT.centerx += x
        foT.centery += y
        self.screen.blit(fo, foT)

    def step(self, moves):
        reward = 0
        self.handleEvents()
        rotated_car_img = self.car.move(moves)

        self.screen.fill("black")

        circuit.circuit(self.screen)
        self.pointLines = circuit.point_lines(self.screen)
        self.pointLines[self.car.currentLine].draw(self.screen)

        hitbox = pygame.draw.rect(self.screen, "orange", self.car.hitbox, 1)

        done = self.checkCircuitCollisions(hitbox)
        reward = self.handleRewards(hitbox)
        self.car.drawRaycasts(self.screen)

        if time.time() - self.starttime > data.time:
            done = True

        self.screen.blit(rotated_car_img, (self.car.x, self.car.y))
        text = self.font.render(str(self.car.points), True, "white", "black")
        textRect = text.get_rect()
        textRect.centerx += 5
        textRect.centery += 5
        times = time.time() - self.starttime
        text2 = self.font.render(str(times.__round__(0)), True, "white", "black")
        textRect2 = text.get_rect()
        textRect2.centerx += 640
        textRect2.centery += 360
        self.screen.blit(text, textRect)
        self.screen.blit(text2, textRect2)
        self.infoLines()

        pygame.display.flip()
        self.clock.tick(60)
        return reward, done, self.car.points

    def handleRewards(self, hitbox):
        lastDistance = self.car.lastDistance
        currentDistance = self.car.currentDistance(
            self.pointLines[self.car.currentLine], True
        )
        reward = 0

        if self.checkPointCollissions(hitbox):
            reward += data.lineReward

        elapsed_time = time.time() - self.starttime
        if elapsed_time > 20:
            reward += data.timeReward

        if currentDistance < lastDistance:
            reward += data.distanceReward * (lastDistance - currentDistance)
        else:
            reward -= data.distancePenalty * (currentDistance - lastDistance)

        if self.checkCircuitCollisions(hitbox):
            reward -= data.hitCost

        if self.car.isMovingInCircles():
            reward -= data.circlePenalty

        self.car.rewardThisGame += reward

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
