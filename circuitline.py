import pygame

class CircuitLine:
    def __init__(self, line: pygame.Rect, start, end):
        self.line = line
        self.start = start
        self.end = end
    def checkCollision(self, rect):
        return bool(self.line.colliderect(rect))