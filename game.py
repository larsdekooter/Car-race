import pygame
from car import Car
import circuit
import time

# import math


class Game:
    no_of_actions = 4
    stateSize = 15

    def __init__(self):
        self.time = 0
        self.points = 0

    def start(self):
        pygame.init()

        self.screen = pygame.display.set_mode((1280, 720))
        self.clock = pygame.time.Clock()
        car = Car()
        self.car = car
        self.starttime = time.time()
        self.font = pygame.font.Font("arial.ttf", 32)
        self.edges = circuit.circuit(self.screen)
        self.car.draw_raycastlines(self.screen)
        return True

    def new_episode(self):
        self.car.reset()

    def step(
        self,
        moves: list,
    ):
        reward = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        rotated_car_img = self.car.move(moves)

        self.screen.fill("black")

        # Box Circuit
        circuit.circuit(self.screen)
        # Point lines
        succeslines = circuit.point_lines(self.screen)

        hitbox = pygame.draw.rect(
            self.screen,
            "orange",
            self.car.hitbox,
            1,
        )
        for edge in self.edges:
            if bool(hitbox.clipline(edge.start, edge.end)):
                self.time = time.time() - self.starttime
                self.points = self.car.points
                # pygame.quit()
                reward = -10
                return reward, True, self.car.points

        for line in succeslines:
            if hitbox.clipline(line.start, line.end):
                index = line.i
                if self.car.lastline == index:
                    pass
                else:
                    self.car.lastline = index
                    self.car.points += 1
                    executionTime = time.time() - self.starttime
                    self.car.times.append(executionTime)
                    reward = 10 + self.points / 13

        self.screen.blit(rotated_car_img, (self.car.x, self.car.y))
        text = self.font.render(
            str(self.car.points) + " " + str(-self.car.speed), True, "white", "black"
        )
        text_rect = text.get_rect()
        text_rect.centerx += 5
        text_rect.centery += 5
        times = time.time() - self.starttime
        text2 = self.font.render(str((times).__round__(0)), True, "white", "black")
        if times > 5:
            reward += 1
        if times > 10:
            reward += 1
        if times > 15:
            reward += 1
        if times > 20:
            reward += 1
        text_rect2 = text.get_rect()
        text_rect2.centerx += 640
        text_rect2.centery += 360

        # for i in self.get_intersections():
        #     for j in i:
        #         pygame.draw.circle(self.screen, "blue", j, 5)

        self.car.draw_raycastlines(self.screen)
        self.screen.blit(text, text_rect)
        self.screen.blit(text2, text_rect2)
        pygame.display.flip()
        self.clock.tick(60)
        return reward, False, self.car.points

    def get_intersections(self):
        inters = []
        for cast in self.car.raycastlines:
            intersections = cast.get_collision_point(self.edges)
            inters.append(intersections)
        return inters

    def reset(self):
        self.car.reset()
        self.starttime = time.time()
