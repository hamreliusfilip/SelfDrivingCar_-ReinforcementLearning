import pygame
import sys
import math
import numpy as np

# Q-learning parameters
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.1  # Exploration rate
NUM_ACTIONS = 4  # Number of actions: left, right, accelerate, decelerate

# Initialize the Q-table (discretize state space if necessary)
# For simplicity, let's assume the car's (x, y) coordinates and angle are divided into discrete bins.
STATE_BINS = (305, 340, 60)  # (x, y, angle bins)
Q_table = np.zeros(STATE_BINS + (NUM_ACTIONS,))  # Create a Q-table

def discretize_state(car_x, car_y, car_angle):
    # Discretize the state by dividing the continuous variables into bins
    x_bin = min(int(car_x / (SCREEN_WIDTH / STATE_BINS[0])), STATE_BINS[0] - 1)
    y_bin = min(int(car_y / (SCREEN_HEIGHT / STATE_BINS[1])), STATE_BINS[1] - 1)
    angle_bin = int(car_angle / 360 * STATE_BINS[2]) % STATE_BINS[2]
    return (x_bin, y_bin, angle_bin)

def choose_action(state):
    if np.random.uniform(0, 1) < EPSILON:
        return np.random.randint(NUM_ACTIONS)  # Explore
    else:
        return np.argmax(Q_table[state])  # Exploit

def update_q_table(state, action, reward, next_state):
    # Q-learning update rule
    best_next_action = np.argmax(Q_table[next_state])
    td_target = reward + GAMMA * Q_table[next_state][best_next_action]
    td_error = td_target - Q_table[state][action]
    Q_table[state][action] += ALPHA * td_error

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
car_width = 20
car_height = 5
car_x = 305
car_y = 340
car_speed = 0  
max_speed = 10  
acceleration = 0.1  
deceleration = 0.05  
rotation_speed = 5  
car_angle = 60

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

    state = discretize_state(car_x, car_y, car_angle)

    while running:
        # Choose an action (0: left, 1: right, 2: accelerate, 3: decelerate)
        action = choose_action(state)

        # Apply the action to the car (steering and speed)
        if action == 0:
            car_angle += rotation_speed
        elif action == 1:
            car_angle -= rotation_speed
        elif action == 2:
            car_speed = min(car_speed + acceleration, max_speed)
        elif action == 3:
            car_speed = max(car_speed - acceleration, -max_speed)

        # Update the car's position
        car_x += car_speed * math.cos(math.radians(-car_angle))
        car_y += car_speed * math.sin(math.radians(-car_angle))

        # Fill the screen with the track image
        screen.blit(track_image, (0, 0))

        # Draw the car and get its bounding rect
        car_rect = draw_car(car_x, car_y, car_angle)

        # Check for collision with the track
        collision = check_collision(car_rect)
        if collision:
            reward = -1  # Punish for hitting the wall
            car_speed = 0  # Stop the car on collision
        else:
            reward = 1  # Reward for staying on the track

        # Get the new state after the action
        next_state = discretize_state(car_x, car_y, car_angle)

        # Update the Q-table
        update_q_table(state, action, reward, next_state)

        # Set the current state to the next state
        state = next_state

        # Update the display
        pygame.display.flip()

        # Set the frame rate
        clock.tick(60)

    pygame.quit()
    sys.exit()

game_loop()
