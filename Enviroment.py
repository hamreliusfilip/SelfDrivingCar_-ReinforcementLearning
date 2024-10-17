import pygame
from settings import *
from bus import Bus

class BusEnvironment:
    def __init__(self, screen, bus):
        self.screen = screen
        self.bus = bus  # Store the bus object
        track = pygame.image.load('final_track.png')
        self.image = pygame.transform.scale(track, (WIDTH, HEIGHT))
        self.contour_points = BARRIERS
        self.checkpoints = CHECKPOINTS

        # List to keep track of whether each checkpoint has been passed
        self.checkpoint_passed = [False] * len(self.checkpoints)

        self.VIS_BARRIERS = True
        self.VIS_CHECKPOINTS = True
        
    def draw(self):
        self.screen.blit(self.image, (0, 0))
       
        # if self.VIS_BARRIERS:
        #     self.draw_barriers()
        # if self.VIS_CHECKPOINTS:
        #     self.draw_checkpoints()
        
        # Draw the bus on the screen
        self.bus.update()
        self.bus.draw(self.screen)


        # Check if the bus passed any checkpoints
        self.update_checkpoints()

    def draw_barriers(self):
        pygame.draw.lines(self.screen, (255, 0, 0), True, self.contour_points[0], 5)
        pygame.draw.lines(self.screen, (255, 0, 0), True, self.contour_points[1], 5)
        for point in np.vstack(self.contour_points):
            pygame.draw.circle(self.screen, (0, 0, 255), point, 5)
            
    def draw_checkpoints(self):
        for index, checkpoint in enumerate(self.checkpoints):
            # Determine the color based on whether the checkpoint has been passed
            if index == 0:
                # Start checkpoint
                pygame.draw.line(self.screen, (0, 255, 0), (checkpoint[0], checkpoint[1]), (checkpoint[2], checkpoint[3]), 3)
                self.screen.blit(pygame.font.Font(None,30).render(f"START", True, (0,255,0)), ((checkpoint[0] - 30, checkpoint[1] - 50)))
            else:
                color = (255, 255, 0) if self.checkpoint_passed[index] else (255, 0, 0)
                pygame.draw.line(self.screen, color, (checkpoint[0], checkpoint[1]), (checkpoint[2], checkpoint[3]), 3)

    def update_checkpoints(self):
        """Check if the bus passed a checkpoint and update the list."""
        checkpoint_threshold = 50  # Threshold distance to consider a checkpoint as "passed"
        bus_position = pygame.math.Vector2(self.bus.rect.center)

        # Loop through checkpoints and check if the bus passed any
        for index, checkpoint in enumerate(self.checkpoints):
            if not self.checkpoint_passed[index]:  # Only check checkpoints not yet passed
                checkpoint_position = pygame.math.Vector2((checkpoint[0] + checkpoint[2]) // 2, (checkpoint[1] + checkpoint[3]) // 2)
                if bus_position.distance_to(checkpoint_position) < checkpoint_threshold:
                    self.checkpoint_passed[index] = True
                    if index == len(self.checkpoints) - 1:
                        self.reset_checkpoints()   # Reset after last checkpoint
                        self.bus.laps_completed += 1

    def reset_checkpoints(self):
        """Reset all checkpoints to the unpassed state (all red)."""
        self.checkpoint_passed = [False] * len(self.checkpoints)
