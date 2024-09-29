import math


def findIntersections(raycast, lines):
    intersections = []
    x1, y1 = raycast.x, raycast.y
    x2, y2 = raycast.end

    for line in lines:
        x3, y3 = line.start
        x4, y4 = line.end

        # Calculate the direction vectors
        rayDir = (x2 - x1, y2 - y1)
        lineDir = (x4 - x3, y4 - y3)

        # Calculate the determinant
        det = rayDir[0] * lineDir[1] - rayDir[1] * lineDir[0]

        if det != 0:
            t = ((x3 - x1) * lineDir[1] - (y3 - y1) * lineDir[0]) / det
            u = ((x3 - x1) * rayDir[1] - (y3 - y1) * rayDir[0]) / det

            if 0 <= t <= 1 and 0 <= u <= 1:
                # Intersection point lies on both lines
                intersection_x = x1 + t * rayDir[0]
                intersection_y = y1 + t * rayDir[1]
                intersections.append((intersection_x, intersection_y))

    return intersections


def getDistanceToPoint(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def getDistanceToLine(x, y, line):
    x1, y1 = line.start
    x2, y2 = line.end

    return abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / math.sqrt(
        (y2 - y1) ** 2 + (x2 - x1) ** 2
    )


def getAngleToLine(x, y, angle, line):
    x1, y1 = line.start
    x2, y2 = line.end

    # Calculate the angle of the line
    line_angle = math.atan2(y2 - y1, x2 - x1)

    # Calculate the angle from the point to the line's midpoint
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    point_to_line_angle = math.atan2(mid_y - y, mid_x - x)

    # Calculate the difference between the angles
    angle_diff = point_to_line_angle - math.radians(angle)

    # Normalize the angle difference to be between -pi and pi
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

    return math.degrees(angle_diff)
