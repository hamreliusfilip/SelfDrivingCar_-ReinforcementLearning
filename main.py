# main.py
import pygame
from settings import *
from Enviroment import BusEnvironment
import neat
from bus import Bus
import pickle

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
      
        bus = Bus(genome, config, random=False)
        env = BusEnvironment(screen, bus)
        score = 0
        done = False

        while not done:
            env.draw()
            bus.update()

            score += bus.compute_fitness()

            if bus.has_collided:
                done = True
                #print(f"Bus collided. Final score: {score}")

            if bus.finished:
                done = True
                print(f"Bus completed the track! Final score: {score}")

                with open(f"successful_genome_{genome_id}.pkl", "wb") as f:
                    pickle.dump(genome, f)
                print(f"Genome {genome_id} saved successfully.")
            pygame.display.flip()

        print(f"Genome {genome_id} scored: {score}")
        genome.fitness = score



def run_neat(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    winner = population.run(eval_genomes, 50)
    
    with open("winner-genome.pkl", "wb") as f:
        pickle.dump(winner, f)


if __name__ == '__main__':

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Autonomous Car - NEAT')

    config_path = 'neat-conf.txt'
    run_neat(config_path)
    pygame.quit()
