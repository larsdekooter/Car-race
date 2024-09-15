import pygame
from circuitline import CircuitLine
from succesline import SuccesLine
import numpy as np


def circuit():
    return [
        CircuitLine("white", (120, 700), (1100, 700)),
        CircuitLine("white", (120, 700), (60, 650)),
        CircuitLine("white", (60, 650), (40, 400)),
        CircuitLine("white", (40, 400), (60, 300)),
        CircuitLine("white", (60, 300), (100, 250)),
        CircuitLine("white", (100, 250), (250, 200)),
        CircuitLine("white", (250, 200), (600, 225)),
        CircuitLine("white", (600, 225), (650, 275)),
        CircuitLine("white", (650, 275), (650, 350)),
        CircuitLine("white", (650, 350), (600, 400)),
        CircuitLine("white", (600, 400), (550, 475)),
        CircuitLine("white", (550, 475), (800, 450)),
        CircuitLine("white", (800, 450), (950, 350)),
        CircuitLine("white", (950, 350), (1050, 325)),
        CircuitLine("white", (1050, 325), (1100, 350)),
        CircuitLine("white", (1100, 350), (1200, 475)),
        CircuitLine("white", (1200, 475), (1175, 600)),
        CircuitLine("white", (1175, 600), (1100, 700)),
        # Inner half
        CircuitLine("white", (170, 650), (900, 650)),
        CircuitLine("white", (170, 650), (110, 600)),
        CircuitLine("white", (110, 600), (90, 450)),
        CircuitLine("white", (90, 450), (110, 350)),
        CircuitLine("white", (110, 350), (150, 300)),
        CircuitLine("white", (150, 300), (250, 250)),
        CircuitLine("white", (250, 250), (550, 275)),
        CircuitLine("white", (550, 275), (575, 300)),
        CircuitLine("white", (575, 300), (475, 525)),
        CircuitLine("white", (475, 525), (800, 500)),
        CircuitLine("white", (800, 500), (950, 400)),
        CircuitLine("white", (950, 400), (1000, 410)),
        CircuitLine("white", (1000, 410), (1000, 600)),
        CircuitLine("white", (1000, 600), (900, 650)),
    ]
    # return [line0]


def point_lines():
    return np.flip(
        [
            SuccesLine(22, "green", (55, 599), (108, 583)),
            SuccesLine(21, "green", (200, 699), (213, 652)),
            SuccesLine(20, "green", (419, 698), (438, 650)),
            SuccesLine(19, "green", (641, 699), (631, 652)),
            SuccesLine(18, "green", (871, 700), (882, 650)),
            SuccesLine(17, "green", (968, 632), (1130, 659)),
            SuccesLine(16, "green", (999, 580), (1181, 580)),
            SuccesLine(15, "green", (1000, 516), (1189, 530)),
            SuccesLine(14, "green", (1000, 454), (1159, 425)),
            SuccesLine(13, "green", (999, 408), (1102, 348)),
            SuccesLine(12, "green", (951, 403), (949, 352)),
            SuccesLine(11, "green", (838, 470), (830, 427)),
            SuccesLine(10, "green", (708, 460), (701, 507)),
            SuccesLine(9, "green", (561, 517), (568, 475)),
            SuccesLine(8, "green", (512, 443), (560, 457)),
            SuccesLine(7, "green", (573, 303), (635, 259)),
            SuccesLine(6, "green", (469, 265), (474, 218)),
            SuccesLine(5, "green", (262, 247), (276, 203)),
            SuccesLine(4, "green", (161, 293), (117, 241)),
            SuccesLine(3, "green", (55, 320), (140, 320)),
            SuccesLine(2, "green", (50, 350), (117, 350)),
            SuccesLine(1, "green", (42, 429), (93, 432)),
            SuccesLine(0, "green", (52, 550), (103, 553)),
        ]
    )


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
            3,
            lambda x: pygame.draw.line(screen, "green", (20, 540), (20 + gap, 540)),
            (20, 540),
            (20 + gap, 540),
        ),
        SuccesLine(
            2,
            lambda x: pygame.draw.line(screen, "green", (20, 350), (20 + gap, 350)),
            (20, 350),
            (20 + gap, 350),
        ),
        SuccesLine(
            1,
            lambda x: pygame.draw.line(screen, "green", (20, 140), (20 + gap, 140)),
            (20, 140),
            (20 + gap, 140),
        ),
        SuccesLine(
            12,
            lambda x: pygame.draw.line(screen, "green", (200, 20), (200, 20 + gap)),
            (200, 20),
            (200, 20 + gap),
        ),
        SuccesLine(
            11,
            lambda x: pygame.draw.line(screen, "green", (600, 20), (600, 20 + gap)),
            (600, 20),
            (600, 20 + gap),
        ),
        SuccesLine(
            10,
            lambda x: pygame.draw.line(screen, "green", (1100, 20), (1100, 20 + gap)),
            (1100, 20),
            (1100, 20 + gap),
        ),
        SuccesLine(
            4,
            lambda x: pygame.draw.line(screen, "green", (1260, 140), (1260 - gap, 140)),
            (1260, 140),
            (1260 - gap, 140),
        ),
        SuccesLine(
            5,
            lambda x: pygame.draw.line(screen, "green", (1260, 350), (1260 - gap, 350)),
            (1260, 350),
            (1260 - gap, 350),
        ),
        SuccesLine(
            6,
            lambda x: pygame.draw.line(screen, "green", (1260, 540), (1260 - gap, 540)),
            (1260, 540),
            (1260 - gap, 540),
        ),
        SuccesLine(
            7,
            lambda x: pygame.draw.line(screen, "green", (1100, 700), (1100, 700 - gap)),
            (1100, 700),
            (1100, 700 - gap),
        ),
        SuccesLine(
            8,
            lambda x: pygame.draw.line(screen, "green", (600, 700), (600, 700 - gap)),
            (600, 700),
            (600, 700 - gap),
        ),
        SuccesLine(
            9,
            lambda x: pygame.draw.line(screen, "green", (200, 700), (200, 700 - gap)),
            (200, 700),
            (200, 700 - gap),
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
