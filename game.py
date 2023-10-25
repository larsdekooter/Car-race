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
        self.edges = circuit.circuit2(self.screen)  # circuit.circuit2(self.screen)
        self.car.draw_raycastlines(self.screen)
        return True

    def new_episode(self):
        self.car.reset()

    def step(
        self,
        moves: list,
    ):
        reward = 0
        self.handle_events()
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         pygame.quit()
        #         exit()

        rotated_car_img = self.car.move(moves)

        self.screen.fill("black")

        # Box Circuit
        circuit.circuit2(self.screen)
        # Point lines
        self.point_lines = circuit.point_lines2(self.screen)

        hitbox = pygame.draw.rect(
            self.screen,
            "orange",
            self.car.hitbox,
            1,
        )
        reward, done, self.car.points = self.check_collisions(hitbox, reward)
        if done:
            return reward, done, self.car.points

        reward = self.check_points(reward, self.point_lines, hitbox)

        reward = self.handle_reward(reward)

        self.screen.blit(rotated_car_img, (self.car.x, self.car.y))
        text = self.font.render(str(self.car.points), True, "white", "black")
        text_rect = text.get_rect()
        text_rect.centerx += 5
        text_rect.centery += 5
        times = time.time() - self.starttime
        text2 = self.font.render(str((times).__round__(0)), True, "white", "black")
        text_rect2 = text.get_rect()
        text_rect2.centerx += 640
        text_rect2.centery += 360
        self.car.draw_raycastlines(self.screen, 0)
        self.screen.blit(text, text_rect)
        self.screen.blit(text2, text_rect2)
        # self.draw_intersections()

        pygame.display.flip()
        self.clock.tick(60)

        if times > 50 and self.car.points < 10:
            reward -= 10
            return reward, True, self.car.points

        return reward, False, self.car.points

    def get_intersections(self):
        inters = []
        for cast in self.car.raycastlines:
            intersections = cast.get_collision_point(self.edges)
            inters.append(intersections)
        return inters

    def get_point_intersections(self):
        inters = []
        for cast in self.car.raycastlines:
            intersections = cast.get_collision_point(self.point_lines)
            inters.append(intersections)
        return inters

    def reset(self):
        self.car.reset()
        self.starttime = time.time()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    def check_collisions(self, hitbox, reward):
        for edge in self.edges:
            if bool(hitbox.clipline(edge.start, edge.end)):
                self.time = time.time() - self.starttime
                self.points = self.car.points
                # pygame.quit()
                reward = -10
                return reward, True, self.car.points
        return reward, False, self.car.points

    def check_points(self, reward, succeslines, hitbox):
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
                    reward = 10  # + self.points / 13 + self.car.d / 10
                    return reward
        return reward

    def handle_reward(self, reward=0):
        if self.time > 20:
            reward += 0.001
        if self.time > 50:
            reward += 0.01

        reward += self.points
        return reward

    def draw_intersections(self):
        intersections = self.get_intersections()
        for line in intersections:
            for intersection in line:
                pygame.draw.circle(self.screen, "white", intersection, 5)
        pointline_intersections = self.get_point_intersections()
        for line in pointline_intersections:
            for intersection in line:
                pygame.draw.circle(self.screen, "green", intersection, 5)
