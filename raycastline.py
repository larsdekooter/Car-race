import pygame
from circuitline import CircuitLine
from succesline import SuccesLine
import data


class RaycastLine:
    def __init__(self, x, y, direction, width=0):
        self.x = x
        self.y = y
        length = data.raycastLength
        end_point = (x + direction[0] * length, y + direction[1] * length)
        self.endpoint = end_point
        self.width = width

    def draw(self, screen):
        pygame.draw.line(screen, "white", (self.x, self.y), self.endpoint, self.width)

    def get_collision_points(self, lines: list[CircuitLine] | list[SuccesLine]):
        return find_intersection(raycast=self, circuit_lines=lines)


def find_intersection(raycast, circuit_lines):
    intersections = []
    x1, y1 = raycast.x, raycast.y
    x2, y2 = raycast.endpoint

    for circuit_line in circuit_lines:
        x3, y3 = circuit_line.start
        x4, y4 = circuit_line.end

        # Calculate the direction vectors
        ray_dir = (x2 - x1, y2 - y1)
        circuit_dir = (x4 - x3, y4 - y3)

        # Calculate the determinant
        det = ray_dir[0] * circuit_dir[1] - ray_dir[1] * circuit_dir[0]

        if det != 0:
            t = ((x3 - x1) * circuit_dir[1] - (y3 - y1) * circuit_dir[0]) / det
            u = ((x3 - x1) * ray_dir[1] - (y3 - y1) * ray_dir[0]) / det

            if 0 <= t <= 1 and 0 <= u <= 1:
                # Intersection point lies on both lines
                intersection_x = x1 + t * ray_dir[0]
                intersection_y = y1 + t * ray_dir[1]
                intersections.append((intersection_x, intersection_y))

    return intersections
