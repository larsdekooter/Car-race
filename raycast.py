import data
import pygame
from circuitline import CircuitLine
from pointline import PointLine


class Raycast:
    def __init__(self, x, y, screen, direction) -> None:
        self.x = x
        self.y = y
        length = data.raycastlength
        self.endPoint = (x + direction[0] * length, y + direction[1] * length)
        self.line = pygame.draw.line(screen, "red", (x, y), self.endPoint, data.width)

    def get_collision_points(self, lines: list[CircuitLine] | list[PointLine]):
        intersections = []
        x1, y1 = self.x, self.y
        x2, y2 = self.endPoint

        for line in lines:
            x3, y3 = line.start
            x4, y4 = line.end

            raydir = (x2 - x1, y2 - y1)
            circuitdir = (x4 - x3, y4 - y3)

            det = raydir[0] * circuitdir[1] - raydir[1] * circuitdir[0]
            if det != 0:
                t = ((x3 - x1) * circuitdir[1] - (y3 - y1) * circuitdir[0]) / det
                u = ((x3 - x1) * raydir[1] - (y3 - y1) * raydir[0]) / det

                if 0 <= t <= 1 and 0 <= u <= 1:
                    intersections.append((x1 + t * raydir[0], y1 + t * raydir[1]))
        return intersections
