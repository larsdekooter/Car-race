import pygame
from circuitline import CircuitLine
from succesline import SuccesLine


class Functions:
    def f1(x):
        return (x, 600)

    def f2(y):
        return (140, y)

    def f3(x):
        return (x, 25 / 140 * x + 525)

    def f4(x):
        return (x, -3 * x + 620.356)


def circuit(screen):
    return [
        CircuitLine(
            pygame.draw.line(screen, "white", (120, 700), (1100, 700)),
            (120, 700),
            (1100, 700),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (120, 700), (60, 650)),
            (120, 700),
            (60, 650),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (60, 650), (40, 400)),
            (60, 650),
            (40, 400),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (40, 400), (60, 300)),
            (40, 400),
            (60, 300),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (60, 300), (100, 250)),
            (60, 300),
            (100, 250),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (100, 250), (250, 200)),
            (100, 250),
            (250, 200),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (250, 200), (600, 225)),
            (250, 200),
            (600, 225),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (600, 225), (650, 275)),
            (600, 225),
            (650, 275),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (650, 275), (650, 350)),
            (650, 275),
            (650, 350),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (650, 350), (600, 400)),
            (650, 350),
            (600, 400),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (600, 400), (550, 475)),
            (600, 400),
            (550, 475),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (550, 475), (800, 450)),
            (550, 475),
            (800, 450),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (800, 450), (950, 350)),
            (800, 450),
            (950, 350),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (950, 350), (1050, 325)),
            (950, 350),
            (1050, 325),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (1050, 325), (1100, 350)),
            (1050, 325),
            (1100, 350),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (1100, 350), (1200, 475)),
            (1100, 350),
            (1200, 475),
        ),  # this one?/
        CircuitLine(
            pygame.draw.line(screen, "white", (1200, 475), (1175, 600)),
            (1200, 475),
            (1175, 600),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (1175, 600), (1100, 700)),
            (1175, 600),
            (1100, 700),
        ),
        # Inner half
        CircuitLine(
            pygame.draw.line(screen, "white", (170, 650), (900, 650)),
            (170, 650),
            (950, 650),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (170, 650), (110, 600)),
            (170, 650),
            (110, 600),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (110, 600), (90, 450)),
            (110, 600),
            (90, 450),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (90, 450), (110, 350)),
            (90, 450),
            (110, 350),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (110, 350), (150, 300)),
            (110, 350),
            (150, 300),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (150, 300), (250, 250)),
            (150, 300),
            (250, 250),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (250, 250), (550, 275)),
            (250, 250),
            (550, 275),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (550, 275), (575, 300)),
            (550, 275),
            (575, 300),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (575, 300), (475, 525)),
            (575, 300),
            (475, 525),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (475, 525), (800, 500)),
            (475, 525),
            (800, 500),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (800, 500), (950, 400)),
            (800, 500),
            (950, 400),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (950, 400), (1000, 410)),
            (950, 400),
            (1000, 410),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (1000, 410), (1000, 600)),
            (1000, 410),
            (1000, 600),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (1000, 600), (900, 650)),
            (1000, 600),
            (950, 650),
        ),
    ]
    # return [line0]


def point_lines(screen):
    return [
        SuccesLine(
            1,
            pygame.draw.line(screen, "green", (999, 408), (1102, 348)),
            (999, 408),
            (1102, 348),
        ),
        SuccesLine(
            2,
            pygame.draw.line(screen, "green", (1000, 454), (1159, 425)),
            (1000, 454),
            (1159, 425),
        ),
        SuccesLine(
            3,
            pygame.draw.line(screen, "green", (951, 403), (949, 352)),
            (951, 403),
            (949, 352),
        ),
        SuccesLine(
            4,
            pygame.draw.line(screen, "green", (838, 470), (830, 427)),
            (838, 470),
            (830, 427),
        ),
        SuccesLine(
            5,
            pygame.draw.line(screen, "green", (708, 460), (701, 507)),
            (708, 460),
            (701, 507),
        ),
        SuccesLine(
            6,
            pygame.draw.line(screen, "green", (561, 517), (568, 475)),
            (561, 517),
            (568, 475),
        ),
        SuccesLine(
            7,
            pygame.draw.line(screen, "green", (512, 443), (560, 457)),
            (512, 443),
            (560, 457),
        ),
        SuccesLine(
            8,
            pygame.draw.line(screen, "green", (573, 303), (635, 259)),
            (573, 303),
            (635, 259),
        ),
        SuccesLine(
            9,
            pygame.draw.line(screen, "green", (469, 265), (474, 218)),
            (469, 265),
            (474, 218),
        ),
        SuccesLine(
            10,
            pygame.draw.line(screen, "green", (262, 247), (276, 203)),
            (262, 247),
            (276, 203),
        ),
        SuccesLine(
            11,
            pygame.draw.line(screen, "green", (161, 293), (117, 241)),
            (161, 293),
            (117, 241),
        ),
        SuccesLine(
            12,
            pygame.draw.line(screen, "green", (42, 429), (93, 432)),
            (42, 429),
            (93, 432),
        ),
        SuccesLine(
            13,
            pygame.draw.line(screen, "green", (55, 599), (108, 583)),
            (55, 599),
            (108, 583),
        ),
        SuccesLine(
            14,
            pygame.draw.line(screen, "green", (200, 699), (213, 652)),
            (200, 699),
            (213, 652),
        ),
        SuccesLine(
            15,
            pygame.draw.line(screen, "green", (419, 698), (438, 650)),
            (419, 698),
            (438, 650),
        ),
        SuccesLine(
            16,
            pygame.draw.line(screen, "green", (641, 699), (631, 652)),
            (641, 699),
            (631, 652),
        ),
        SuccesLine(
            17,
            pygame.draw.line(screen, "green", (871, 700), (882, 650)),
            (871, 700),
            (882, 650),
        ),
        SuccesLine(
            17,
            pygame.draw.line(screen, "green", (999, 580), (1181, 580)),
            (999, 580),
            (1181, 580),
        ),
        SuccesLine(
            17,
            pygame.draw.line(screen, "green", (968, 632), (1130, 659)),
            (968, 632),
            (1130, 659),
        ),
    ]


gap = 100


def circuit2(screen):
    return [
        CircuitLine(
            pygame.draw.line(screen, "white", (20, 700), (1260, 700)),
            (20, 700),
            (1260, 700),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (1260, 700), (1260, 20)),
            (1260, 700),
            (1260, 20),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (1260, 20), (20, 20)),
            (1260, 20),
            (20, 20),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (20, 20), (20, 700)),
            (20, 20),
            (20, 700),
        ),
        # inner part
        CircuitLine(
            pygame.draw.line(
                screen, "white", (20 + gap, 700 - gap), (1260 - gap, 700 - gap)
            ),
            (20 + gap, 700 - gap),
            (1260 - gap, 700 - gap),
        ),
        CircuitLine(
            pygame.draw.line(
                screen, "white", (1260 - gap, 700 - gap), (1260 - gap, 20 + gap)
            ),
            (1260 - gap, 700 - gap),
            (1260 - gap, 20 + gap),
        ),
        CircuitLine(
            pygame.draw.line(
                screen, "white", (1260 - gap, 20 + gap), (20 + gap, 20 + gap)
            ),
            (1260 - gap, 20 + gap),
            (20 + gap, 20 + gap),
        ),
        CircuitLine(
            pygame.draw.line(
                screen, "white", (20 + gap, 20 + gap), (20 + gap, 700 - gap)
            ),
            (20 + gap, 20 + gap),
            (20 + gap, 700 - gap),
        ),
    ]


def point_lines2(screen):
    return [
        SuccesLine(
            1,
            pygame.draw.line(screen, "green", (20, 350), (20 + gap, 350)),
            (20, 350),
            (20 + gap, 350),
        ),
        SuccesLine(
            2,
            pygame.draw.line(screen, "green", (20, 140), (20 + gap, 140)),
            (20, 140),
            (20 + gap, 140),
        ),
        SuccesLine(
            3,
            pygame.draw.line(screen, "green", (20, 540), (20 + gap, 540)),
            (20, 540),
            (20 + gap, 540),
        ),
        SuccesLine(
            4,
            pygame.draw.line(screen, "green", (1260, 140), (1260 - gap, 140)),
            (1260, 140),
            (1260 - gap, 140),
        ),
        SuccesLine(
            5,
            pygame.draw.line(screen, "green", (1260, 350), (1260 - gap, 350)),
            (1260, 350),
            (1260 - gap, 350),
        ),
        SuccesLine(
            6,
            pygame.draw.line(screen, "green", (1260, 540), (1260 - gap, 540)),
            (1260, 540),
            (1260 - gap, 540),
        ),
        SuccesLine(
            7,
            pygame.draw.line(screen, "green", (1100, 700), (1100, 700 - gap)),
            (1100, 700),
            (1100, 700 - gap),
        ),
        SuccesLine(
            8,
            pygame.draw.line(screen, "green", (600, 700), (600, 700 - gap)),
            (600, 700),
            (600, 700 - gap),
        ),
        SuccesLine(
            9,
            pygame.draw.line(screen, "green", (200, 700), (200, 700 - gap)),
            (200, 700),
            (200, 700 - gap),
        ),
        SuccesLine(
            10,
            pygame.draw.line(screen, "green", (1100, 20), (1100, 20 + gap)),
            (1100, 20),
            (1100, 20 + gap),
        ),
        SuccesLine(
            11,
            pygame.draw.line(screen, "green", (600, 20), (600, 20 + gap)),
            (600, 20),
            (600, 20 + gap),
        ),
        SuccesLine(
            12,
            pygame.draw.line(screen, "green", (200, 20), (200, 20 + gap)),
            (200, 20),
            (200, 20 + gap),
        ),
    ]


def circuit3(screen):
    return [
        CircuitLine(
            pygame.draw.line(screen, "white", (1270, 710), (1270, 10)),
            (1270, 710),
            (1270, 10),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (1270, 10), (10, 10)),
            (1270, 10),
            (10, 10),
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (10, 10), (10, 710)), (10, 10), (10, 710)
        ),
        CircuitLine(
            pygame.draw.line(screen, "white", (1270, 710), (10, 710)),
            (1270, 710),
            (0, 710),
        ),
    ]
