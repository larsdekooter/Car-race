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
