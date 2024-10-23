import pygame
from settings import *
from bus import Bus

class BusEnvironment:
    def __init__(self, screen, bus):
        self.screen = screen
        self.bus = bus  
        track = pygame.image.load('final_track.png')
        self.image = pygame.transform.scale(track, (WIDTH, HEIGHT))
        self.contour_points = BARRIERS
        self.checkpoints = CHECKPOINTS

        self.checkpoint_passed = [False] * len(self.checkpoints)

        self.VIS_BARRIERS = True
        self.VIS_CHECKPOINTS = True
        
        self.happy_smiley = pygame.image.load('happy.png')  
        self.sad_smiley = pygame.image.load('sad.png')    
        self.smiley_size = (65, 65) 
        
    def draw(self):
        self.screen.blit(self.image, (0, 0))
       
       # Uncomment to see checkpoints
       
        # if self.VIS_BARRIERS:
        #     self.draw_barriers()
        # if self.VIS_CHECKPOINTS:
        #     self.draw_checkpoints()
        
        speed_text = pygame.font.Font(None, 30).render(f"Fart: {12.5 * self.bus.velocity:.2f} km/h", True, (0, 0, 0))
        self.screen.blit(speed_text, (WIDTH - 170, 10))  
        
        self.bus.update()
        self.bus.draw(self.screen)
        # self.draw_smiley()

        self.update_checkpoints()
        
    def draw_smiley(self):
      
        box_width = 120
        box_height = 100
        box_pos_x = WIDTH - box_width - 10  
        box_pos_y = 60  

        border_color = (0, 0, 0)  
        fill_color = (200, 200, 200)  

        radius = 10  
        pygame.draw.rect(self.screen, fill_color, (box_pos_x, box_pos_y, box_width, box_height), 0, radius)  
        pygame.draw.rect(self.screen, border_color, (box_pos_x, box_pos_y, box_width, box_height), 5, radius)

        font = pygame.font.Font(None, 30)
        text = font.render("Din Fart", True, (0, 0, 0))  
        text_rect = text.get_rect(center=(box_pos_x + box_width // 2, box_pos_y + 20)) 
        self.screen.blit(text, text_rect)

        if self.bus.schoolzone[self.bus.checkpoint_index] == 1: 
            if self.bus.velocity < 2:  
                smiley_image = pygame.transform.scale(self.happy_smiley, self.smiley_size)  
            else: 
                smiley_image = pygame.transform.scale(self.sad_smiley, self.smiley_size)  
        else:
            if self.bus.velocity > 2: 
                smiley_image = pygame.transform.scale(self.happy_smiley, self.smiley_size)  
            else:  
                smiley_image = pygame.transform.scale(self.sad_smiley, self.smiley_size) 

        smiley_pos_x = box_pos_x + (box_width - self.smiley_size[0]) // 2  
        smiley_pos_y = box_pos_y + (box_height - self.smiley_size[1]) // 2 + 10 

        self.screen.blit(smiley_image, (smiley_pos_x, smiley_pos_y))

    def draw_barriers(self):
        pygame.draw.lines(self.screen, (255, 0, 0), True, self.contour_points[0], 5)
        pygame.draw.lines(self.screen, (255, 0, 0), True, self.contour_points[1], 5)
        for point in np.vstack(self.contour_points):
            pygame.draw.circle(self.screen, (0, 0, 255), point, 5)
            
    def draw_checkpoints(self):
        for index, checkpoint in enumerate(self.checkpoints):
     
            if index == 0:
                pygame.draw.line(self.screen, (0, 255, 0), (checkpoint[0], checkpoint[1]), (checkpoint[2], checkpoint[3]), 3)
                self.screen.blit(pygame.font.Font(None,30).render(f"START", True, (0,255,0)), ((checkpoint[0] - 30, checkpoint[1] - 50)))
            else:
                color = (255, 255, 0) if self.checkpoint_passed[index] else (255, 0, 0)
                pygame.draw.line(self.screen, color, (checkpoint[0], checkpoint[1]), (checkpoint[2], checkpoint[3]), 3)

    def update_checkpoints(self):
        """Check if the bus passed a checkpoint and update the list."""
        checkpoint_threshold = 50 
        bus_position = pygame.math.Vector2(self.bus.rect.center)

        for index, checkpoint in enumerate(self.checkpoints):
            if not self.checkpoint_passed[index]:  
                checkpoint_position = pygame.math.Vector2((checkpoint[0] + checkpoint[2]) // 2, (checkpoint[1] + checkpoint[3]) // 2)
                if bus_position.distance_to(checkpoint_position) < checkpoint_threshold:
                    self.checkpoint_passed[index] = True
                    if index == len(self.checkpoints) - 1:
                        self.reset_checkpoints()  
                        self.bus.laps_completed += 1

    def reset_checkpoints(self):
        """Reset all checkpoints to the unpassed state (all red)."""
        self.checkpoint_passed = [False] * len(self.checkpoints)
