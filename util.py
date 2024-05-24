import math
import numpy as np
from succesline import SuccesLine
from raycastline import RaycastLine


def getDistanceToPoint(x, y, point):
    dx = x - point[0]
    dy = y - point[1]
    return math.sqrt(dx * dx + dy * dy)


def getDistanceToLine(x, y, line: SuccesLine, raycasts: list[RaycastLine]):
    distances = []
    for raycast in raycasts:
        points = raycast.get_collision_points([line])
        if len(points) > 0:
            distance = getDistanceToPoint(x, y, points[0])
            distances.append(distance)
    return np.min(distances) if len(distances) > 1 else 1000


def getShortestDistanceToLine(x, y, line: SuccesLine):
    point = np.asarray([x, y])
    line_start = np.asarray(line.start)
    line_end = np.asarray(line.end)
    line_direction = line_end - line_start
    point_line_vector = point - line_start
    line_length = np.linalg.norm(line_direction)
    t = np.dot(point_line_vector, line_direction) / np.dot(
        line_direction, line_direction
    )
    if t < 0 or t > 1:
        projection = (
            line_start
            if np.linalg.norm(point_line_vector) < np.linalg.norm(point - line_end)
            else line_end
        )
    else:
        projection = line_start + t * line_direction
    distance_vector = point - projection
    distance = np.linalg.norm(distance_vector)
    return distance, distance_vector[0], distance_vector[1]
