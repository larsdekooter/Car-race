import pygame


pygame.init()

screen = pygame.display.set_mode((1280, 720))


clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            print(pygame.mouse.get_pos())

    # pygame.draw.line(screen, "white", (120, 700), (1100, 700))
    # pygame.draw.line(screen, "white", (120, 700), (60, 650))
    # pygame.draw.line(screen, "white", (60, 650), (40, 400))
    # pygame.draw.line(screen, "white", (40, 400), (60, 300))
    # pygame.draw.line(screen, "white", (60, 300), (100, 250))
    # pygame.draw.line(screen, "white", (100, 250), (250, 200))
    # pygame.draw.line(screen, "white", (250, 200), (600, 225))
    # pygame.draw.line(screen, "white", (600, 225), (650, 275))
    # pygame.draw.line(screen, "white", (650, 275), (650, 350))
    # pygame.draw.line(screen, "white", (650, 350), (600, 400))
    # pygame.draw.line(screen, "white", (600, 400), (550, 475))
    # pygame.draw.line(screen, "white", (550, 475), (800, 450))
    # pygame.draw.line(screen, "white", (800, 450), (950, 350))
    # pygame.draw.line(screen, "white", (950, 350), (1050, 325))
    # pygame.draw.line(screen, "white", (1050, 325), (1100, 350))
    # pygame.draw.line(screen, "white", (1100, 350), (1200, 475))
    # pygame.draw.line(screen, "white", (1200, 475), (1175, 600))
    # pygame.draw.line(screen, "white", (1175, 600), (1100, 700))
    # pygame.draw.line(screen, "white", (170, 650), (950, 650))
    # pygame.draw.line(screen, "white", (170, 650), (110, 600))
    # pygame.draw.line(screen, "white", (110, 600), (90, 450))
    # pygame.draw.line(screen, "white", (90, 450), (110, 350))
    # pygame.draw.line(screen, "white", (110, 350), (150, 300))
    # pygame.draw.line(screen, "white", (150, 300), (250, 250))
    # pygame.draw.line(screen, "white", (250, 250), (550, 275))
    # pygame.draw.line(screen, "white", (550, 275), (575, 300))
    # pygame.draw.line(screen, "white", (575, 300), (475, 525))
    # pygame.draw.line(screen, "white", (475, 525), (800, 500))
    # pygame.draw.line(screen, "white", (800, 500), (950, 400))
    # pygame.draw.line(screen, "white", (950, 400), (1000, 410))
    # pygame.draw.line(screen, "white", (1000, 410), (1000, 600))
    # pygame.draw.line(screen, "white", (1000, 600), (950, 650))

    # (pygame.draw.line(screen, "green", (999, 408), (1102, 348)))
    # (pygame.draw.line(screen, "green", (1000, 454), (1159, 425)))
    # (pygame.draw.line(screen, "green", (951, 403), (949, 352)))
    # (pygame.draw.line(screen, "green", (838, 470), (830, 427)))
    # (pygame.draw.line(screen, "green", (708, 460), (701, 507)))
    # (pygame.draw.line(screen, "green", (561, 517), (568, 475)))
    # (pygame.draw.line(screen, "green", (512, 443), (560, 457)))
    # (pygame.draw.line(screen, "green", (573, 303), (635, 259)))
    # (pygame.draw.line(screen, "green", (469, 265), (474, 218)))
    # (pygame.draw.line(screen, "green", (262, 247), (276, 203)))
    # (pygame.draw.line(screen, "green", (161, 293), (117, 241)))
    # (pygame.draw.line(screen, "green", (42, 429), (93, 432)))
    # (pygame.draw.line(screen, "green", (55, 599), (108, 583)))
    # (pygame.draw.line(screen, "green", (200, 699), (213, 652)))
    # (pygame.draw.line(screen, "green", (419, 698), (438, 650)))
    # (pygame.draw.line(screen, "green", (641, 699), (631, 652)))
    # (pygame.draw.line(screen, "green", (871, 700), (882, 650)))
    gap = 100
    pygame.draw.line(screen, "white", (20, 700), (1260, 700)),
    pygame.draw.line(screen, "white", (1260, 700), (1260, 20)),
    pygame.draw.line(screen, "white", (1260, 20), (20, 20)),
    pygame.draw.line(screen, "white", (20, 20), (20, 700)),
    # inner part
    pygame.draw.line(screen, "white", (20 + gap, 700 - gap), (1260 - gap, 700 - gap)),
    pygame.draw.line(screen, "white", (1260 - gap, 700 - gap), (1260 - gap, 20 + gap)),
    pygame.draw.line(screen, "white", (1260 - gap, 20 + gap), (20 + gap, 20 + gap)),
    pygame.draw.line(screen, "white", (20 + gap, 20 + gap), (20 + gap, 700 - gap)),

    pygame.draw.line(screen, "green", (20, 350), (20 + gap, 350))
    pygame.draw.line(screen, "green", (20, 140), (20 + gap, 140))
    pygame.draw.line(screen, "green", (20, 540), (20 + gap, 540))
    pygame.draw.line(screen, "green", (1260, 140), (1260 - gap, 140))
    pygame.draw.line(screen, "green", (1260, 350), (1260 - gap, 350))
    pygame.draw.line(screen, "green", (1260, 540), (1260 - gap, 540))
    pygame.draw.line(screen, "green", (1100, 700), (1100, 700 - gap))
    pygame.draw.line(screen, "green", (600, 700), (600, 700 - gap))
    pygame.draw.line(screen, "green", (200, 700), (200, 700 - gap))
    pygame.draw.line(screen, "green", (1100, 20), (1100, 20 + gap))
    pygame.draw.line(screen, "green", (600, 20), (600, 20 + gap))
    pygame.draw.line(screen, "green", (200, 20), (200, 20 + gap))

    pygame.display.update()
    screen.fill("black")
    clock.tick(60)
