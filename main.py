# main.py
import pygame
from settings import *
from Enviroment import BusEnvironment
import neat
from bus import Bus
import pickle

# Function to evaluate genomes
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        # Create the bus controlled by NEAT for this genome
        bus = Bus(genome, config)  # Create the bus

        # Set up environment with this bus
        env = BusEnvironment(screen, bus)  # Pass the bus to the environment

        score = 0
        done = False

        # Run the simulation loop for the bus controlled by this genome
        while not done:
            env.draw()  # Draw the environment, including the bus
            bus.update()  # Update the bus (NEAT will decide the action)

            # Example fitness calculation: increment score for every successful step
            score += bus.compute_fitness()

            # Stop the simulation if a collision occurs or the bus finishes a lap
            if bus.has_collided:
                done = True
                print(f"Bus collided. Final score: {score}")
            
            # Check if the bus completed the track
            if bus.finished:
                done = True
                print(f"Bus completed the track! Final score: {score}")

                # Save the genome when the bus completes the track
                with open(f"successful_genome_{genome_id}.pkl", "wb") as f:
                    pickle.dump(genome, f)
                print(f"Genome {genome_id} saved successfully.")

            pygame.display.flip()
            pygame.time.delay(20)

        # Assign the final score to the genome's fitness
        genome.fitness = score

# Main function to run NEAT
def run_neat(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    # Create the NEAT population
    population = neat.Population(config)

    # Add a reporter to track progress
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Run NEAT (train over generations)
    winner = population.run(eval_genomes, 10)
    with open("winner-genome.pkl", "wb") as f:
        pickle.dump(winner, f)

if __name__ == '__main__':
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Autonomous Car - NEAT')

    # Path to the NEAT config file
    config_path = 'neat-conf.txt'
    run_neat(config_path)

    pygame.quit()

