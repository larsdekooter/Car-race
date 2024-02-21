from circuitline import CircuitLine
import pygame
from pointline import PointLine

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

def pointLines(screen):
    return [
        PointLine(
            1,
            lambda x: pygame.draw.line(screen, "green", (1000, 516), (1189, 530)),
            (1000, 516),
            (1189, 530),
        ),
        PointLine(
            2,
            lambda x: pygame.draw.line(screen, "green", (1000, 454), (1159, 425)),
            (1000, 454),
            (1159, 425),
        ),
        PointLine(
            3,
            lambda x: pygame.draw.line(screen, "green", (999, 408), (1102, 348)),
            (999, 408),
            (1102, 348),
        ),
        PointLine(
            4,
            lambda x: pygame.draw.line(screen, "green", (951, 403), (949, 352)),
            (951, 403),
            (949, 352),
        ),
        PointLine(
            5,
            lambda x: pygame.draw.line(screen, "green", (838, 470), (830, 427)),
            (838, 470),
            (830, 427),
        ),
        PointLine(
            6,
            lambda x: pygame.draw.line(screen, "green", (708, 460), (701, 507)),
            (708, 460),
            (701, 507),
        ),
        PointLine(
            7,
            lambda x: pygame.draw.line(screen, "green", (561, 517), (568, 475)),
            (561, 517),
            (568, 475),
        ),
        PointLine(
            8,
            lambda x: pygame.draw.line(screen, "green", (512, 443), (560, 457)),
            (512, 443),
            (560, 457),
        ),
        PointLine(
            9,
            lambda x: pygame.draw.line(screen, "green", (573, 303), (635, 259)),
            (573, 303),
            (635, 259),
        ),
        PointLine(
            10,
            lambda x: pygame.draw.line(screen, "green", (469, 265), (474, 218)),
            (469, 265),
            (474, 218),
        ),
        PointLine(
            11,
            lambda x: pygame.draw.line(screen, "green", (262, 247), (276, 203)),
            (262, 247),
            (276, 203),
        ),
        PointLine(
            12,
            lambda x: pygame.draw.line(screen, "green", (161, 293), (117, 241)),
            (161, 293),
            (117, 241),
        ),
        PointLine(
            13,
            lambda x: pygame.draw.line(screen, "green", (42, 429), (93, 432)),
            (42, 429),
            (93, 432),
        ),
        PointLine(
            14,
            lambda x: pygame.draw.line(screen, "green", (55, 599), (108, 583)),
            (55, 599),
            (108, 583),
        ),
        PointLine(
            15,
            lambda x: pygame.draw.line(screen, "green", (200, 699), (213, 652)),
            (200, 699),
            (213, 652),
        ),
        PointLine(
            16,
            lambda x: pygame.draw.line(screen, "green", (419, 698), (438, 650)),
            (419, 698),
            (438, 650),
        ),
        PointLine(
            17,
            lambda x: pygame.draw.line(screen, "green", (641, 699), (631, 652)),
            (641, 699),
            (631, 652),
        ),
        PointLine(
            18,
            lambda x: pygame.draw.line(screen, "green", (871, 700), (882, 650)),
            (871, 700),
            (882, 650),
        ),
        PointLine(
            19,
            lambda x: pygame.draw.line(screen, "green", (968, 632), (1130, 659)),
            (968, 632),
            (1130, 659),
        ),
        PointLine(
            20,
            lambda x: pygame.draw.line(screen, "green", (999, 580), (1181, 580)),
            (999, 580),
            (1181, 580),
        ),
    ]