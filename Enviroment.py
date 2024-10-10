# enviroment.py
import pygame
from settings import *
from bus import Bus

class BusEnvironment:
    def __init__(self, screen, bus):
        self.screen = screen
        self.bus = bus  # Store the bus object
        track = pygame.image.load('new_track.png')
        self.image = pygame.transform.scale(track, (WIDTH, HEIGHT))
        self.contour_points = BARRIERS
        self.checkpoints = CHECKPOINTS

        self.VIS_BARRIERS = True
        self.VIS_CHECKPOINTS = True

    def draw(self):
        self.screen.blit(self.image, (0, 0))

        # Optionally display text (e.g., number of laps completed)
        text = pygame.font.Font(None, 30).render(f"Laps completed: 0", True, (0, 0, 0))
        self.screen.blit(text, (10, 10))  # Display text at top-left corner

        if self.VIS_BARRIERS:
            self.draw_barriers()
        if self.VIS_CHECKPOINTS:
            self.draw_checkpoints()
        
        # Draw the bus on the screen
        self.bus.update()
        self.bus.draw(self.screen)


    def draw_barriers(self):
        pygame.draw.lines(self.screen, (255, 0, 0), True, self.contour_points[0], 5)
        pygame.draw.lines(self.screen, (255, 0, 0), True, self.contour_points[1], 5)
        for point in np.vstack(self.contour_points):
            pygame.draw.circle(self.screen, (0, 0, 255), point, 5)

    def draw_checkpoints(self):
        counter = 0
        for checkpoint in self.checkpoints:
            if counter == 0:
                pygame.draw.line(self.screen, (0, 255, 0), (checkpoint[0], checkpoint[1]), (checkpoint[2], checkpoint[3]), 3)
                self.screen.blit(pygame.font.Font(None,30).render(f"START", True, (0,255,0)), ((checkpoint[0] - 30, checkpoint[1] - 50)))
            else:
                pygame.draw.line(self.screen, (255, 0, 0), (checkpoint[0], checkpoint[1]), (checkpoint[2], checkpoint[3]), 3)
            counter += 1
