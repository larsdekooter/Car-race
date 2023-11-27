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
    x1, y1 = line.start
    x2, y2 = line.end

    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    dotProduct = ((x - x1) * (x2 - x1)) + ((y - y1) * (y2 - y1))
    projection = dotProduct / length**2
    point = (x1 + (projection * (x2 - x1)), y1 + (projection * (y2 - y1)))
    distance = math.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
    return distance
