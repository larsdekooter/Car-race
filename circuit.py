import pygame
import numpy as np


class CircuitLine:
    def __init__(self, color, start: tuple[int, int], end: tuple[int, int]):
        self.start = start
        self.end = end
        self.color = color

    def draw(self, screen):
        pygame.draw.line(screen, self.color, self.start, self.end)


class PointLine:
    def __init__(self, index: int, color, start: tuple[int, int], end: tuple[int, int]):
        self.start = start
        self.end = end
        self.index = index
        self.color = color

    def draw(self, screen):
        pygame.draw.line(screen, self.color, self.start, self.end)


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


def pointLines():
    return np.flip(
        [
            PointLine(22, "green", (55, 599), (108, 583)),
            PointLine(21, "green", (200, 699), (213, 652)),
            PointLine(20, "green", (419, 698), (438, 650)),
            PointLine(19, "green", (641, 699), (631, 652)),
            PointLine(18, "green", (871, 700), (882, 650)),
            PointLine(17, "green", (968, 632), (1130, 659)),
            PointLine(16, "green", (999, 580), (1181, 580)),
            PointLine(15, "green", (1000, 516), (1189, 530)),
            PointLine(14, "green", (1000, 454), (1159, 425)),
            PointLine(13, "green", (999, 408), (1102, 348)),
            PointLine(12, "green", (951, 403), (949, 352)),
            PointLine(11, "green", (838, 470), (830, 427)),
            PointLine(10, "green", (708, 460), (701, 507)),
            PointLine(9, "green", (561, 517), (568, 475)),
            PointLine(8, "green", (512, 443), (560, 457)),
            PointLine(7, "green", (573, 303), (635, 259)),
            PointLine(6, "green", (469, 265), (474, 218)),
            PointLine(5, "green", (262, 247), (276, 203)),
            PointLine(4, "green", (161, 293), (117, 241)),
            PointLine(3, "green", (55, 320), (140, 320)),
            PointLine(2, "green", (50, 350), (117, 350)),
            PointLine(1, "green", (42, 429), (93, 432)),
            PointLine(0, "green", (52, 550), (103, 553)),
        ]
    )
