import pygame
from settings import *
from settings import CHECKPOINTS  # Import checkpoint positions from settings

import sys

# Initialize pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((800, 600))  # Adjust size as necessary
pygame.display.set_caption('Checkpoint Visualizer')

# Load checkpoints from your settings
CHECKPOINTS = [
    (100, 100), (200, 150), (300, 200), (400, 250),  # Replace with real checkpoint positions
    (500, 300), (600, 350), (700, 400)
]

def draw_checkpoints(screen, checkpoints):
    """Draw checkpoints with numbers on the screen."""
    screen.fill((0, 0, 0))  # Fill the background with black

    font = pygame.font.Font(None, 36)  # Set the font size for numbering

    # Draw each checkpoint
    for i, checkpoint in enumerate(checkpoints):
        pygame.draw.circle(screen, (0, 255, 0), checkpoint, 10)  # Draw a green circle for each checkpoint
        text = font.render(str(i), True, (255, 255, 255))  # Create text for checkpoint number
        screen.blit(text, (checkpoint[0] + 10, checkpoint[1]))  # Position text next to the checkpoint

# Main loop to display checkpoints
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Draw checkpoints on the screen
    draw_checkpoints(screen, CHECKPOINTS)

    # Update the display
    pygame.display.flip()

    # Frame rate (Optional)
    pygame.time.Clock().tick(30)

# Quit pygame
pygame.quit()
sys.exit()
