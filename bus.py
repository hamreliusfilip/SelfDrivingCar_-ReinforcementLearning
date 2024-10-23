import pygame
import numpy as np
from settings import *
import neat
import math

class Bus():
    def __init__(self, genome, config, random=False):

        bus_image = pygame.image.load('bus.png')
        self.original_image = pygame.transform.scale(bus_image, (55, 100))
        self.image = self.original_image 
        self.rect = self.image.get_rect()
        self.barriers = BARRIERS  
        self.checkpoints = CHECKPOINTS
        self.schoolzone = SCHOOLZONE
        self.has_collided = False
        self.finished = False
        self.checkpoint_index = 0
        self.checkpoint_reward = 50
        self.laps_completed = 0 

        if random:
            self.get_start_pos_from_checkpoint()
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
        self.position = np.array(self.rect.center, dtype=np.float64)

        self.radar_distances = self.get_radar_distances()
        self.velocity = 4  
        self.rotation_speed = 3 
        self.max_velocity = 4  
        self.acceleration = 0.1 
        self.steering_angle = 0
        self.max_steering_angle = 30 


    def get_start_pos_from_checkpoint(self):
        """Sets the bus to a random checkpoint with correct angle to the next checkpoint."""
        index = np.random.randint(0, len(self.checkpoints) - 1)

        self.bus_start_pos = STARTING_POSITIONS[index]

        next_index = (index + 1) % len(self.checkpoints)
        self.angle = ANGLES[index]
        
        self.checkpoint_index = next_index


    def update(self):
        
        action = self.decide_action()

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
        self.position += np.array([direction.x, direction.y]) * self.velocity
        
        self.rect.center = self.position.astype(int)

    def decide_action(self):

        inputs = [self.radar_distances[0], self.radar_distances[1], self.radar_distances[2], self.radar_distances[3], self.radar_distances[4], SCHOOLZONE[self.checkpoint_index]]
        outputs = self.net.activate(inputs)
        action = outputs.index(max(outputs)) 
        
        return action
    
    def draw(self, screen):
        rotated_image = pygame.transform.rotate(self.original_image, -self.angle)
        rotated_rect = rotated_image.get_rect(center=self.rect.center)
        screen.blit(rotated_image, rotated_rect.topleft)

        self.draw_radar(screen)


    def draw_radar(self, screen):
        center = self.rect.center
        for i, radar_angle in enumerate(self.radar_angles):
            adjusted_angle = self.angle + radar_angle

            ray_direction = pygame.math.Vector2(0, -1).rotate(adjusted_angle)
            radar_end = center + np.array([ray_direction.x, ray_direction.y]) * self.radar_distances[i]

            # Uncomment to draw radar line
            
            # pygame.draw.line(screen, (0, 255, 0), center, radar_end, 2)
            # pygame.draw.circle(screen, (0, 0, 255), radar_end, 5)


    def get_radar_distances(self):
        """Return the distances for each radar ray, only up to the barriers."""
        center = np.array(self.rect.center)
        radar_distances = []
        
        for radar_angle in self.radar_angles:

            ray_direction = pygame.math.Vector2(0, -1).rotate(self.angle + radar_angle)
            radar_end = center + np.array([ray_direction.x, ray_direction.y]) * self.radar_length
            nearest_intersection = self.find_nearest_intersection(center, radar_end)

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
        if denom == 0:
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

        t = t_numer / denom
        intersection_point = p0 + t * s10
        return intersection_point

    def check_for_checkpoints(self):
        """Checks if the back of the bus has crossed a checkpoint line and returns a reward."""
        if self.checkpoint_index > (len(self.checkpoints)-1):
            self.finished = True
            return 0

        checkpoint = self.checkpoints[self.checkpoint_index]
        x1, y1, x2, y2 = checkpoint

        direction = pygame.math.Vector2(0, 1).rotate(self.angle) 
        back_position = np.array(self.rect.center) + np.array([direction.x, direction.y]) * (self.rect.height / 2)

        if self.has_crossed_line(self.previous_position, back_position, np.array([x1, y1]), np.array([x2, y2])):
            self.checkpoint_index += 1
            return self.checkpoint_reward
        return 0


    def has_crossed_line(self, previous_position, current_position, line_start, line_end):
        """Check if a line between previous_position and current_position crosses the checkpoint line."""

        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        d1 = sign(current_position, line_start, line_end)
        d2 = sign(previous_position, line_start, line_end)

        return d1 * d2 < 0

    def compute_fitness(self):
        reward = 0
        distance_moved = np.linalg.norm(self.rect.center - self.previous_position)

        reward += distance_moved * (2 if self.schoolzone[self.checkpoint_index] == 0 else 1)
        
        reward += 0.1 

        if self.schoolzone[self.checkpoint_index] == 1 and self.velocity > 2:
            reward -= 20
            
        if self.schoolzone[self.checkpoint_index] == 1 and self.velocity < 2:
            reward += 100

        if min(self.radar_distances) < 10:
            reward -= 100 
            self.has_collided = True

            if self.schoolzone[self.checkpoint_index] == 1:
                reward -= 50  

       
        checkpoint_reward = self.check_for_checkpoints()
        reward += checkpoint_reward * 2  

        self.previous_position = np.array(self.rect.center)

        return reward

