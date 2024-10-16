import pygame
import neat
import pickle
import numpy as np
from settings import *
from Enviroment import BusEnviroment
from bus import Bus

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Autonomous Bus - NEAT (Trained Model)')
FONT = pygame.font.Font(None, 32)

MAX_STEPS = 500  # Set the maximum number of steps per simulation episode

def run_trained_genome(genome, config):
    # Create the neural network from the loaded genome
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    env = BusEnviroment(screen)
    done = False
    state = env.bus.get_radar_distances()
    state = state / env.bus.radar_length  # Normalize the radar distances

    for step in range(MAX_STEPS):
        if done:
            break

        # Activate the neural network based on the current state
        actions = net.activate(state)

        move_forward = actions[0] > 0.5  # Threshold for moving forward
        turn_left = actions[1] > 0.5     # Threshold for turning left
        turn_right = actions[2] > 0.5    # Threshold for turning right

        env.bus.update_action(move_forward, turn_left, turn_right)

        # Get the next state (normalized radar distances)
        state = env.bus.get_radar_distances() / env.bus.radar_length

        # Check if the bus collided with a wall
        if env.bus.has_collided:
            done = True

        # Update and draw the environment
        env.draw()
        pygame.display.flip()

        pygame.time.delay(50)  # Adjust speed of simulation by delaying frame update

def load_and_run_model(config_file, genome_file):
    # Load the configuration file used for training
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    # Load the trained genome from file
    with open(genome_file, 'rb') as f:
        winner_genome = pickle.load(f)

    # Run the trained genome in the environment
    run_trained_genome(winner_genome, config)

if __name__ == "__main__":
    # Specify the config file and genome file (adjust paths as needed)
    config_path = "./neat-config.txt"
    genome_path = "./winner-genome.pkl"

    # Load and run the model
    load_and_run_model(config_path, genome_path)

    # Quit pygame when done
    pygame.quit()

