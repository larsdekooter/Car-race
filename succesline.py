import pygame


class SuccesLine:
    def __init__(self, i, color, start, end):
        self.color = color
        self.i = i
        self.start = start
        self.end = end
        self.isDrawn = False

    def check_collision(self, rect):
        return bool(self.line.colliderect(rect))

    def draw(self, screen):
        self.isDrawn = True
        return pygame.draw.line(screen, self.color, self.start, self.end)
