import pygame
from car import Car
from circuit import circuit, pointLines


class Game:
    def __init__(self):
        pygame.init()
        self.car = Car()
        self.screen = pygame.display.set_mode((1280, 720))
        self.clock = pygame.time.Clock()
        self.circuit = circuit(self.screen)
        self.pointLines = pointLines(self.screen)

    def step(self, move):
        self.events()
        point, hit = self.basics(move)
        if point:
            self.car.score += 1
        reward = self.reward(point, hit)
        return reward, hit, self.car.score

    def reward(self, point: bool, hit):
        reward = 0
        if point:
            reward += 10
        if hit:
            reward = -10
        return reward

    def basics(self, move):
        self.screen.fill("black")

        carimg = self.car.move(move)

        circuit(self.screen)
        pointLines(self.screen)
        self.pointLines[self.car.currentLine].draw(self.screen)

        hitbox = pygame.draw.rect(self.screen, "orange", self.car.hitbox, 1)
        hit = self.circuitCollision(hitbox)

        point = self.pointCollision(hitbox)
        if point:
            self.car.currentLine += 1

        self.car.drawRaycasts(self.screen)

        self.screen.blit(carimg, (self.car.x, self.car.y))
        pygame.display.flip()
        self.clock.tick(60)
        return point, hit

    def circuitCollision(self, hitbox):
        for line in self.circuit:
            if bool(hitbox.clipline(line.start, line.end)):
                return True

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    def pointCollision(self, hitbox):
        line = self.pointLines[self.car.currentLine]
        return bool(hitbox.clipline(line.start, line.end))
