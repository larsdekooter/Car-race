import pygame


class CircuitLine:
    def __init__(self, color, start, end):

        self.color = color
        self.start = start
        self.end = end

    def check_collision(self, rect):
        return bool(self.line.colliderect(rect))

    def draw(self, screen):
        return pygame.draw.line(screen, self.color, self.start, self.end)
