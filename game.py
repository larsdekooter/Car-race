from car import Car
import pygame
import circuit
import time
import data


class Game:
    def __init__(self):
        self.car = Car()
        pygame.init()
        self.screen = pygame.display.set_mode((1280, 720))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font("arial.ttf", 32)
        self.circuitLines = circuit.circuit(self.screen)
        self.pointLines = circuit.point_lines(self.screen)
        self.starttime = time.time()
        self.car.drawRaycasts(self.screen)
        self.closestDistance = None
        pass

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

        pygame.display.flip()
        self.clock.tick(60)
        return reward, done, self.car.points

    def handleRewards(self, hitbox):
        reward = 0
        if self.checkPointCollissions(hitbox):
            reward += data.lineReward
        if time.time() - self.starttime > 20:
            reward += data.timeReward
        if (
            self.closestDistance != None
            and self.car.lastDistance < self.closestDistance
        ):
            reward += data.distanceReward * (
                self.closestDistance - self.car.lastDistance
            )
            self.closestDistance = self.car.lastDistance
        elif self.closestDistance == None:
            self.closestDistance = self.car.lastDistance
        if self.checkCircuitCollisions(hitbox):
            reward = data.hitCost
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
