import pygame
import sys
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 1024
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Car Driving with Collision Detection")

# Define colors (RGB values)
WHITE = (255, 255, 255)
TOLERANCE = 50

# Car settings
car_width = 40
car_height = 10
car_x = 310
car_y = 340
car_speed = 0  
max_speed = 10  
acceleration = 0.1  
deceleration = 0.05  
rotation_speed = 5  
car_angle = -60

# Load car image and track image
car_image = pygame.image.load('Filip_Test/car.png')
car_image = pygame.transform.rotate(car_image, 180)

car_image = pygame.transform.scale(car_image, (car_width, car_height))
track_image = pygame.image.load('Filip_Test/track.png')
track_image = pygame.transform.scale(track_image, (SCREEN_WIDTH, SCREEN_HEIGHT))

# Function to rotate and blit the car image
def draw_car(x, y, angle):
    rotated_image = pygame.transform.rotate(car_image, angle)
    new_rect = rotated_image.get_rect(center=car_image.get_rect(topleft=(x, y)).center)
    screen.blit(rotated_image, new_rect.topleft)

    # Return the bounding rectangle of the rotated car
    return new_rect

# Function to check if the pixel color is close to the target color within tolerance
def is_color_close(pixel_color, target_color, tolerance):
    return all(abs(pixel_color[i] - target_color[i]) <= tolerance for i in range(3))

def check_collision(car_rect):

    car_center_x = car_rect.centerx
    car_center_y = car_rect.centery

    pixel_color = track_image.get_at((int(car_center_x), int(car_center_y)))
        
    if not is_color_close(pixel_color, WHITE, TOLERANCE):
        return False  
    else :
        return True
        

def game_loop():
    
    global car_x, car_y, car_speed, car_angle

    clock = pygame.time.Clock()
    running = True

    while running:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get keys pressed
        keys = pygame.key.get_pressed()

        # Steering (rotation)
        if keys[pygame.K_LEFT]:
            car_angle += rotation_speed
        if keys[pygame.K_RIGHT]:
            car_angle -= rotation_speed

        # Forward and backward movement (speed control)
        if keys[pygame.K_UP]:
            car_speed = min(car_speed + acceleration, max_speed)
        elif keys[pygame.K_DOWN]:
            car_speed = max(car_speed - acceleration, -max_speed)
        else:
            # Gradually slow down when no key is pressed
            if car_speed > 0:
                car_speed = max(car_speed - deceleration, 0)
            elif car_speed < 0:
                car_speed = min(car_speed + deceleration, 0)

        # Calculate the car's new position based on its angle and speed
        car_x += car_speed * math.cos(math.radians(-car_angle))
        car_y += car_speed * math.sin(math.radians(-car_angle))

        # Fill the screen with the track image
        screen.blit(track_image, (0, 0))

        # Draw the car with the current angle and get its bounding rect
        car_rect = draw_car(car_x, car_y, car_angle)

        # Check for collision with the track
        if check_collision(car_rect):
            print("Collision detected! Car is off the road.")
            car_speed = 0

        # Update the display
        pygame.display.flip()

        # Set the frame rate
        clock.tick(60)

    pygame.quit()
    sys.exit()

game_loop()
