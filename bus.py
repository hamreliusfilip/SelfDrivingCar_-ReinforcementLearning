import pygame
import numpy as np
from settings import *
import neat

class Bus():
    def __init__(self, genome, config, random=False):
        # Load and scale the bus image
        bus_image = pygame.image.load('bus.png')
        self.original_image = pygame.transform.scale(bus_image, (50, 50))
        self.image = self.original_image  # This will hold the rotated image
        self.rect = self.image.get_rect()
        self.barriers = BARRIERS  # Barriers are contours (polygons)
        self.checkpoints = CHECKPOINTS
        self.has_collided = False
        self.finished = False
        self.checkpoint_index = 0
        self.checkpoint_reward = 50

        # Initialize bus starting position and angle
        if random:
            self.bus_start_pos = None
            self.angle = None
        else:
            self.bus_start_pos = INIT_POS
            self.angle = INIT_ANGLE

        self.rect.center = self.bus_start_pos
        self.previous_position = np.array(self.rect.center)

        self.perceived_points = None
        self.radar_length = 100
        self.num_rays = 5
        self.radar_angles = np.linspace(90, -90, self.num_rays)
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)

        # Initialize radar distances
        self.radar_distances = self.get_radar_distances()

        # Movement attributes
        self.velocity = 0  # Forward/Backward speed
        self.rotation_speed = 3  # Speed of turning
        self.max_velocity = 4  # Max speed limit
        self.acceleration = 0.1  # Acceleration rate

        # Angle change for steering
        self.steering_angle = 0
        self.max_steering_angle = 30  # Max turning angle in degrees

    def get_start_pos_from_checkpoint(self):
        index = np.random.randint(0, len(self.checkpoints))

        self.bus_start_pos = STARTING_POSITIONS[index]
        self.angle = INIT_ANGLE

    def update(self):
        
        action = self.decide_action()

        # Move based on the decided action
        if action == 0:  # Go straight
            self.velocity = min(self.velocity + self.acceleration, self.max_velocity)
            self.steering_angle = 0
        elif action == 1:  # Turn left
            self.velocity = min(self.velocity + self.acceleration, self.max_velocity)
            self.angle += self.rotation_speed * (self.velocity / self.max_velocity)
        elif action == 2:  # Turn right
            self.velocity = min(self.velocity + self.acceleration, self.max_velocity)
            self.angle -= self.rotation_speed * (self.velocity / self.max_velocity)
        elif action == 3:  # Slow down
            self.velocity = max(self.velocity - self.acceleration, 1)
            self.steering_angle = 0

        # Update position
        self.radar_distances = self.get_radar_distances()
        direction = pygame.math.Vector2(0, -1).rotate(self.angle)
        self.rect.center += direction * self.velocity

    def decide_action(self):
        # Radar distances are the input for NEAT
        inputs = self.radar_distances

        # Output will be the decision made by the neural network
        outputs = self.net.activate(inputs)

        # Convert the output into a discrete action (0: go straight, 1: turn left, 2: turn right)
        action = outputs.index(max(outputs))  # Choose the action with the highest output
        return action

    def draw(self, screen):
        # Rotate the bus image and draw it on the screen
        rotated_image = pygame.transform.rotate(self.original_image, self.angle)
        rotated_rect = rotated_image.get_rect(center=self.rect.center)
        screen.blit(rotated_image, rotated_rect.topleft)
        
        self.draw_radar(screen)

    def draw_radar(self, screen):
        center = self.rect.center
        for i, radar_angle in enumerate(self.radar_angles):
            # Adjust radar angle based on the bus's current rotation
            adjusted_angle = self.angle + radar_angle
            
            # Calculate radar ray direction
            ray_direction = pygame.math.Vector2(0, -1).rotate(adjusted_angle)  # Use (0, -1) for upward direction
            radar_end = center + np.array([ray_direction.x, ray_direction.y]) * self.radar_distances[i]
            
            # Draw radar line
            pygame.draw.line(screen, (0, 255, 0), center, radar_end, 2)
            pygame.draw.circle(screen, (0, 0, 255), radar_end, 5)

    def get_radar_distances(self):
        """Return the distances for each radar ray, only up to the barriers."""
        center = np.array(self.rect.center)
        radar_distances = []

        for radar_angle in self.radar_angles:
            # Calculate radar ray direction and end point
            ray_direction = pygame.math.Vector2(0, -1).rotate(self.angle + radar_angle)  # Use (0, -1) for upward direction
            radar_end = center + np.array([ray_direction.x, ray_direction.y]) * self.radar_length
            
            # Find the nearest intersection with the barriers
            nearest_intersection = self.find_nearest_intersection(center, radar_end)

            # Calculate distance to nearest intersection or max radar length
            if nearest_intersection is not None:
                distance = np.linalg.norm(nearest_intersection - center)
            else:
                distance = self.radar_length
            
            radar_distances.append(distance)

        return np.array(radar_distances)

    def find_nearest_intersection(self, ray_start, ray_end):
        """Find the nearest intersection point between the radar ray and the barriers."""
        closest_intersection = None
        min_distance = float('inf')

        for contour in self.barriers:
            for i in range(len(contour)):
                p1 = np.array(contour[i])
                p2 = np.array(contour[(i + 1) % len(contour)])

                intersection = self.line_intersection(ray_start, ray_end, p1, p2)

                if intersection is not None:
                    distance = np.linalg.norm(intersection - ray_start)
                    if distance < min_distance:
                        min_distance = distance
                        closest_intersection = intersection

        return closest_intersection

    def line_intersection(self, p0, p1, p2, p3):
        """Check for the intersection between two lines (p0->p1 and p2->p3) and return the intersection point."""
        s10 = p1 - p0
        s32 = p3 - p2

        denom = s10[0] * s32[1] - s32[0] * s10[1]
        if denom == 0:  # Parallel lines
            return None

        denom_is_positive = denom > 0

        s02 = p0 - p2
        s_numer = s10[0] * s02[1] - s10[1] * s02[0]

        if (s_numer < 0) == denom_is_positive:
            return None

        t_numer = s32[0] * s02[1] - s32[1] * s02[0]
        if (t_numer < 0) == denom_is_positive:
            return None

        if (s_numer > denom) == denom_is_positive or (t_numer > denom) == denom_is_positive:
            return None

        # Intersection detected
        t = t_numer / denom
        intersection_point = p0 + t * s10
        return intersection_point

    def check_for_checkpoints(self):
        """Checks if the bus has passed a checkpoint and returns a reward."""
        if self.checkpoint_index >= len(self.checkpoints):
            self.finished = True
            return 0

        # Extract the x, y position from the checkpoint
        checkpoint = np.array(self.checkpoints[self.checkpoint_index][:2])  # Assuming checkpoint has more than 2 values (x, y, width, height)
        bus_position = np.array(self.rect.center)

        # Define a threshold distance to determine if the bus has passed the checkpoint
        checkpoint_threshold = 50

        distance_to_checkpoint = np.linalg.norm(bus_position - checkpoint)

        if distance_to_checkpoint < checkpoint_threshold:
            # Bus has passed the checkpoint, give reward and move to the next checkpoint
            self.checkpoint_index += 1
            print(f"Checkpoint passed: {self.checkpoint_index}")
            return self.checkpoint_reward

        return 0  # No checkpoint passed

    def compute_fitness(self):
        reward = 0
        distance_moved = np.linalg.norm(self.rect.center - self.previous_position)
        reward += distance_moved

        if min(self.radar_distances) < 10:  
            reward -= 100 
            self.has_collided = True

        checkpoint_reward = self.check_for_checkpoints()
        reward += checkpoint_reward

        self.previous_position = np.array(self.rect.center)
        
        return reward

