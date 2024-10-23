import pygame
import neat
import pickle
from settings import *
from Enviroment import BusEnvironment
from bus import Bus

def draw_button(screen, text, x, y, width, height, inactive_color, active_color, font_size):

    pygame.draw.rect(screen, inactive_color, (x, y, width, height))
    
    font = pygame.font.Font(None, font_size)
    text_surf = font.render(text, True, (0, 0, 0))
    text_rect = text_surf.get_rect(center=((x + width // 2), (y + height // 2)))
    screen.blit(text_surf, text_rect)

def check_button(x, y, width, height, prev_click):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    
    if x + width > mouse[0] > x and y + height > mouse[1] > y:
     
        if click[0] == 1 and not prev_click:
            return True, True  
        elif click[0] == 0:
            return False, False 

    return False, prev_click  

def run_saved_genome(genome_file, config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    with open(genome_file, "rb") as f:
        genome = pickle.load(f)

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    simulating = True

    bus = None
    env = None

    button_x = WIDTH - 150
    button_y = HEIGHT - 50
    button_x_pause = WIDTH - 300
    button_y_pause = HEIGHT - 50
    button_width = 130
    button_height = 40
    button_inactive_color = (255, 255, 255)
    button_active_color = (255, 255, 255)

    prev_click_quit = False
    prev_click_pause = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if simulating:
            if bus is None or env is None:
                bus = Bus(genome, config, random=False)
                env = BusEnvironment(screen, bus)

            env.draw()
            bus.update()

        draw_button(screen, 'Quit', button_x, button_y, button_width, button_height, button_inactive_color, button_active_color, 30)
        quit_clicked, prev_click_quit = check_button(button_x, button_y, button_width, button_height, prev_click_quit)
        if quit_clicked:
            print("Quit button clicked!")
            running = False

        draw_button(screen, 'Pause', button_x_pause, button_y_pause, button_width, button_height, button_inactive_color, button_active_color, 30)
        pause_clicked, prev_click_pause = check_button(button_x_pause, button_y_pause, button_width, button_height, prev_click_pause)
        if pause_clicked:
            print("Pause button clicked!")
            simulating = not simulating  

        pygame.display.flip()
        clock.tick(120)

if __name__ == '__main__':
    pygame.init()
    pygame.display.set_caption('Self Driving Bus')

    config_path = 'neat-conf.txt'
    genome_path = 'successful_genome_808.pkl'
    
    run_saved_genome(genome_path, config_path)

    pygame.quit()
