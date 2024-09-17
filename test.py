import pygame
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Pygame Initialization
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1100, 900  # Set the size of the window
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Self-Driving Car Simulation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Car parameters
CAR_WIDTH = 80  # Width of the car
CAR_HEIGHT = 40  # Height of the car
CAR_SPEED = 1  # Speed of the car

# Neural Network Parameters
INPUT_DIM = 8  # Number of input features for the neural network
HIDDEN_DIM = 64  # Number of hidden units in the neural network
OUTPUT_DIM = 3  # Number of possible actions (accelerate, turn left, turn right)

# Font initialization (global variable)
FONT = pygame.font.Font(None, 36)  # You can adjust the font size

# Neural Network Model - Q-Learning
class DQN(nn.Module):
    """
    Deep Q-Network for approximating the Q-value function.
    Consists of three fully connected layers.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation
        x = torch.relu(self.fc2(x))  # Apply ReLU activation
        x = self.fc3(x)  # Output the Q-values for each action
        return x

# Car class
class Car:
    """
    Represents the car in the simulation. Manages its position, movement, and image rendering.
    """
    def __init__(self):
        # Initialize the car's position, angle, and velocity
        self.x = 120 + CAR_WIDTH // 2
        self.y = 120 + CAR_HEIGHT // 2
        self.angle = 0
        self.velocity = 0
        
        # Initialize the car's checkpoint
        self.current_checkpoint = 0  # Start at the first checkpoint

        # Load and initialize the car image
        self.car_image = pygame.image.load("Mclaren.png").convert_alpha()
        self.car_image = pygame.transform.scale(self.car_image, (CAR_WIDTH, CAR_HEIGHT))
        self.rotated_image = self.car_image
        self.rect = self.car_image.get_rect(center=(self.x, self.y))

    def move(self, action):
        """
        Move the car based on the given action.
        Action 0: Accelerate, Action 1: Turn left, Action 2: Turn right
        """
        if action == 0:  # Accelerate
            self.velocity = CAR_SPEED
        elif action == 1:  # Turn left
            self.angle -= 5
        elif action == 2:  # Turn right
            self.angle += 5

        # Update the car's position based on its velocity and angle
        self.x += self.velocity * np.cos(np.radians(self.angle))
        self.y += self.velocity * np.sin(np.radians(self.angle))

        # Rotate the car image and update its rectangle
        self.rotated_image = pygame.transform.rotate(self.car_image, -self.angle)
        self.rect = self.rotated_image.get_rect(center=(self.x, self.y))

    def draw(self):
        """
        Draw the car on the screen.
        """
        SCREEN.blit(self.rotated_image, self.rect.topleft)

    def get_state(self):
        state = np.array([
            self.x / WIDTH,  # Normalize x position
            self.y / HEIGHT,  # Normalize y position
            self.velocity / CAR_SPEED,  # Normalize velocity
            self.angle / 360,  # Normalize angle (assuming it ranges from 0 to 360 degrees)
            self.x / WIDTH,  # Normalize distance to left wall (same as x position)
            (WIDTH - self.x) / WIDTH,  # Normalize distance to right wall
            self.y / HEIGHT,  # Normalize distance to top wall (same as y position)
            (HEIGHT - self.y) / HEIGHT  # Normalize distance to bottom wall
        ])
        return state

    def is_collision(self):
        """
        Check if the car goes off the track.
        The car should stay on the L-shaped track (made of two rectangular areas).
        """
        # Define the L-shaped track boundaries (using two main areas)
        # Top horizontal part (x: 100 to 700, y: 100 to 200)
        horizontal_track = pygame.Rect(100, 100, 600, 100)
        
        # Right vertical part (x: 660 to 760, y: 100 to 700)
        vertical_track = pygame.Rect(660, 100, 100, 600)
        
        # Check if the car is within the bounds of either the horizontal or vertical track
        car_rect = pygame.Rect(self.rect.topleft, (CAR_WIDTH, CAR_HEIGHT))
        
        # If the car is not inside any of the track areas, it's off the track (collision)
        if not horizontal_track.colliderect(car_rect) and not vertical_track.colliderect(car_rect):
            return True  # Collision or off-track detected
        
        return False  # No collision, the car is on the track

def draw_track():
    """
    Draw the L-shaped track and checkpoints on the screen.
    """
    pygame.draw.rect(SCREEN, BLACK, pygame.Rect(100, 100, 600, 100))  # Top horizontal part
    pygame.draw.rect(SCREEN, BLACK, pygame.Rect(660, 100, 100, 600))  # Right vertical part

    # Draw checkpoints
    checkpoints = [
        pygame.Rect(200, 100, 10, 100),
        pygame.Rect(400, 100, 10, 100),
        pygame.Rect(600, 100, 10, 100),
        pygame.Rect(660, 200, 100, 10),
        pygame.Rect(660, 300, 100, 10),
        pygame.Rect(660, 400, 100, 10),
        pygame.Rect(660, 500, 100, 10),
        pygame.Rect(660, 600, 100, 10),
    ]

    for checkpoint in checkpoints:
        pygame.draw.rect(SCREEN, GREEN, checkpoint)

def calculate_reward(car, checkpoints):
    """
    Calculate the reward based on the car's position and interaction with the track.
    This system encourages the car to move toward checkpoints and penalizes for collisions.
    """
    reward = -1  # Small penalty for taking steps (to encourage efficient behavior)
    done = False

    # Check if car has collided
    if car.is_collision():
        reward = -100  # Big penalty for collision
        done = True
    else:
        # Calculate distance to the current target checkpoint
        target_checkpoint = checkpoints[car.current_checkpoint]
        distance_to_checkpoint = np.sqrt((car.x - target_checkpoint.center[0])**2 + (car.y - target_checkpoint.center[1])**2)

        # Reward for moving closer to the target checkpoint
        reward += max(10 - distance_to_checkpoint / 10, 0)  # Reward for approaching the checkpoint

        # Reward for reaching the checkpoint
        if target_checkpoint.collidepoint(car.x, car.y):
            reward += 100  # Big reward for reaching the checkpoint
            car.current_checkpoint += 1  # Move to the next checkpoint in sequence

            # Reset checkpoint sequence if all checkpoints have been passed
            if car.current_checkpoint >= len(checkpoints):
                car.current_checkpoint = 0  # Restart sequence for continuous learning

    return reward, done

# ---------------- PRIORITIZED EXPERIENCE REPLAY ------------------
class SumTree:
    """
    A SumTree data structure to store priorities and quickly sample based on priority for Prioritized Experience Replay.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def add(self, priority, data):
        """
        Add a new priority and experience to the tree.
        """
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        """
        Update the priority of an experience.
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get_leaf(self, s):
        """
        Sample an experience from the tree based on the sum of priorities.
        """
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.tree):
                leaf = parent
                break
            else:
                if s <= self.tree[left]:
                    parent = left
                else:
                    s -= self.tree[left]
                    parent = right
        leaf_idx = leaf
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_priority(self):
        """
        Get the total sum of priorities.
        """
        return self.tree[0]

class PrioritizedReplayBuffer:
    """
    Replay buffer that samples experiences with higher priority more frequently.
    """
    def __init__(self, capacity, alpha):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 0.01  # Small constant to ensure no zero priorities
        self.capacity = capacity

    def add(self, error, sample):
        priority = (abs(error) + self.epsilon) ** self.alpha  # Ensure priority is a non-negative float
        self.tree.add(priority, sample)

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of experiences, weighted by priority.
        """
        minibatch = []
        idxs = []
        segment = self.tree.total_priority() / batch_size
        priorities = []
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get_leaf(s)
            minibatch.append(data)
            idxs.append(idx)
            priorities.append(priority)
        sampling_probabilities = priorities / self.tree.total_priority()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weight /= is_weight.max()
        return minibatch, idxs, is_weight

    def update(self, idx, error):
        """
        Update the priority of a sampled experience.
        """
        priority = (error + self.epsilon) ** self.alpha
        self.tree.update(idx, priority)

# ----------------------- DQN TRAINING AND MEMORY -----------------------
# Replay buffer with prioritized experience replay
MEMORY_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99  # Discount factor
ALPHA = 0.001  # Learning rate
EPSILON = 0.1  # Epsilon for epsilon-greedy action selection
TARGET_UPDATE = 100  # How often to update the target network

# Initialize the networks
policy_net = DQN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
target_net = DQN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Optimizer and loss function
optimizer = optim.Adam(policy_net.parameters(), lr=ALPHA)
criterion = nn.MSELoss()

# Initialize replay buffer
memory = PrioritizedReplayBuffer(MEMORY_SIZE, alpha=0.6)

def remember(state, action, reward, next_state, done):
    """
    Add a new experience to the replay buffer.
    """
    transition = (state, action, reward, next_state, done)
    error = reward  # Initial priority is based on reward
    memory.add(error, transition)

def replay():
    """
    Sample a batch of experiences from the replay buffer and update the Q-network.
    """
    if memory.tree.n_entries < BATCH_SIZE:
        return
    
    minibatch, idxs, is_weights = memory.sample(BATCH_SIZE)
    
    states, actions, rewards, next_states, dones = zip(*minibatch)
    
    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(np.array(actions), dtype=torch.long)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    dones = torch.tensor(np.array(dones), dtype=torch.float32)
    is_weights = torch.tensor(np.array(is_weights), dtype=torch.float32)

    # Compute Q-values
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    max_next_q_values = target_net(next_states).max(1)[0]
    target_q_values = rewards + (GAMMA * max_next_q_values * (1 - dones))

    # Compute the error
    errors = torch.abs(q_values - target_q_values).detach().numpy()

    # Update priorities in the memory
    for i in range(BATCH_SIZE):
        memory.update(idxs[i], errors[i])

    # Compute the loss and backpropagate
    loss = (is_weights * criterion(q_values, target_q_values)).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def act(state):
    """
    Select an action using epsilon-greedy policy.
    """
    if random.random() < EPSILON:
        return random.randrange(OUTPUT_DIM)  # Ensure action is within range
    else:
        state = torch.tensor(np.array([state]), dtype=torch.float32)
        with torch.no_grad():
            return policy_net(state).argmax().item()  # Ensure action is within range

def pre_train():
    global training_phase
    print("Starting pre-training phase")
    car = Car()
    checkpoints = [
        pygame.Rect(200, 100, 10, 100),
        pygame.Rect(400, 100, 10, 100),
        pygame.Rect(600, 100, 10, 100),
        pygame.Rect(660, 200, 100, 10),
        pygame.Rect(660, 300, 100, 10),
        pygame.Rect(660, 400, 100, 10),
        pygame.Rect(660, 500, 100, 10),
        pygame.Rect(660, 600, 100, 10),
    ]
    
    training_phase = False  # Start in pre-training phase

    for _ in range(1000):  # Collect 100 random transitions before training
        state = car.get_state()
        action = random.randrange(OUTPUT_DIM)  # Ensure action is within range
        car.move(action)
        reward, done = calculate_reward(car, checkpoints)
        next_state = car.get_state()
        remember(state, action, reward, next_state, done)
        if done:
            car = Car()  # Reset the car if the episode ends

    training_phase = True  # End of pre-training, start training phase
    print("Pre-training phase completed, starting training phase")

# ----------------------- GAME LOOP -----------------------
def game_loop():
    print("Starting game loop")
    running = True
    clock = pygame.time.Clock()

    car = Car()
    checkpoints = [
        pygame.Rect(200, 100, 10, 100),
        pygame.Rect(400, 100, 10, 100),
        pygame.Rect(600, 100, 10, 100),
        pygame.Rect(660, 200, 100, 10),
        pygame.Rect(660, 300, 100, 10),
        pygame.Rect(660, 400, 100, 10),
        pygame.Rect(660, 500, 100, 10),
        pygame.Rect(660, 600, 100, 10),
    ]

    step_count = 0

    while running:
        SCREEN.fill(WHITE)
        draw_track()

        state = car.get_state()
        action = act(state)
        car.move(action)
        reward, done = calculate_reward(car, checkpoints)

        if car.is_collision():
            done = True
            car = Car()  # Reset the car to the starting position

        next_state = car.get_state()
        remember(state, action, reward, next_state, done)
        replay()

        if step_count % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())  # Update target network

        car.draw()

        # Display the current phase
        phase_text = "Training" if training_phase else "Pre-training"
        phase_surface = FONT.render(phase_text, True, BLACK)
        SCREEN.blit(phase_surface, (10, 10))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        step_count += 1
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    pre_train()  # Execute pre-training phase
    game_loop()  # Start the game loop

