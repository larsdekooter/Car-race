import pygame
import math
from raycastline import RaycastLine

X, Y = 1208, 496
print(X, Y)


class Car:
    def __init__(self):
        self.x = X
        self.y = Y
        self.speed = 0
        self.angle = 0
        self.max_speed = 5
        self.acceleration = 0.2
        self.decelaration = 0.5
        self.turn_speed = 10
        self.img = pygame.transform.scale(pygame.image.load("car.png"), (25, 25))
        self.rect = self.img.get_rect()
        self.hitbox = self.update_hitbox()
        self.points = 0
        self.lastline = -1
        self.times = []
        self.d = 0

    def move(self, dirs: list):
        if dirs[1] == 1:
            self.speed += self.acceleration
            if self.speed > self.max_speed:
                self.speed = self.max_speed
        elif dirs[0] == 1:
            self.speed -= self.decelaration
            if self.speed < -self.max_speed:
                self.speed = -self.max_speed
        else:
            if self.speed > 0:
                self.speed -= self.decelaration
            elif self.speed < 0:
                self.speed += self.decelaration
            if abs(self.speed) < self.decelaration:
                self.speed = 0

        if dirs[2] == 1:
            self.angle += self.turn_speed
            if self.angle >= 360:
                leftover = self.angle - 360
                self.angle = leftover
        elif dirs[3] == 1:
            self.angle -= self.turn_speed
            if self.angle <= -360:
                leftover = self.angle + 360  # angle is -361
                self.angle = leftover

        radians = self.angle * (math.pi / 180)
        x_change = self.speed * math.sin(radians)
        y_change = self.speed * math.cos(radians)

        self.x += x_change
        self.y += y_change
        self.d += x_change + y_change
        self.update_hitbox()

        return pygame.transform.rotate(self.img, self.angle)

    def get_direction(self, keys):
        dirs = []
        if keys[pygame.K_UP]:
            dirs.append("UP")
        if keys[pygame.K_DOWN]:
            dirs.append("DOWN")
        if keys[pygame.K_LEFT]:
            dirs.append("LEFT")
        if keys[pygame.K_RIGHT]:
            dirs.append("RIGHT")
        return dirs

    def update_hitbox(self):
        self.hitbox = (self.x, self.y, 20, 20)

    def copy(self):
        return self

    def reset(self):
        self.x = X
        self.y = Y
        self.speed = 0
        self.angle = 0
        self.max_speed = 5
        self.acceleration = 0.2
        self.decelaration = 0.5
        self.turn_speed = 10
        self.img = pygame.transform.scale(pygame.image.load("car.png"), (25, 25))
        self.rect = self.img.get_rect()
        self.hitbox = self.update_hitbox()
        self.points = 0
        self.lastline = -1
        self.times = []
        self.d = 0

    def get_looking_direction(self, angle):
        angle_radians = math.radians(angle)
        return (math.sin(angle_radians), math.cos(angle_radians))

    def draw_raycastlines(
        self,
        screen,
    ):
        self.raycastlines = [
            RaycastLine(
                self.x + 20, self.y + 20, screen, self.get_looking_direction(self.angle)
            ),
            RaycastLine(
                self.x + 20,
                self.y + 20,
                screen,
                (
                    -self.get_looking_direction(self.angle)[0],
                    -self.get_looking_direction(self.angle)[1],
                ),
            ),
            RaycastLine(
                self.x + 20,
                self.y + 20,
                screen,
                self.get_looking_direction(self.angle + 90),
            ),
            RaycastLine(
                self.x + 20,
                self.y + 20,
                screen,
                self.get_looking_direction(self.angle - 90),
            ),
            RaycastLine(
                self.x + 20,
                self.y + 20,
                screen,
                self.get_looking_direction(self.angle - 45),
            ),
            RaycastLine(
                self.x + 20,
                self.y + 20,
                screen,
                self.get_looking_direction(self.angle + 45),
            ),
            RaycastLine(
                self.x + 20,
                self.y + 20,
                screen,
                (
                    -self.get_looking_direction(self.angle - 45)[0],
                    -self.get_looking_direction(self.angle - 45)[1],
                ),
            ),
            RaycastLine(
                self.x + 20,
                self.y + 20,
                screen,
                (
                    -self.get_looking_direction(self.angle + 45)[0],
                    -self.get_looking_direction(self.angle + 45)[1],
                ),
            ),
        ]

    def get_distance_to(self, point):
        x = point[0]
        y = point[1]
        dx = x - self.x
        dy = y - self.y
        return math.sqrt(dx * dx + dy * dy)
