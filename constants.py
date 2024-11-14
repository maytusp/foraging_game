import pygame

cell_size = 50  # Size of each grid cell in pixels



# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
AGENT_COLOR = (0, 255, 0)  # Green
HOME_COLOR = (173, 216, 230)  # Light blue
FOOD_COLORS = [(255, 0, 0), (255, 165, 0), (128, 0, 128), (165, 42, 42), (255, 192, 203)]  # Red, Orange, Purple, Brown, Pink
# Define controls
AGENT1_KEYS = {pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d, pygame.K_q, pygame.K_e}
AGENT2_KEYS = {pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_o, pygame.K_p}

# Load and scale the robot images
agent1_image = pygame.transform.scale(pygame.image.load("./figs/agents/agent1.png"), (cell_size, cell_size))
agent2_image = pygame.transform.scale(pygame.image.load("./figs/agents/agent2.png"), (cell_size, cell_size))

agent_images = [agent1_image, agent2_image]


# Load and scale the food images
spinach =  pygame.transform.scale(pygame.image.load("./figs/foods/spinach.png"), (cell_size, cell_size))
watermelon =  pygame.transform.scale(pygame.image.load("./figs/foods/watermelon.png"), (cell_size, cell_size))
strawberry =  pygame.transform.scale(pygame.image.load("./figs/foods/strawberry.png"), (cell_size, cell_size))
chicken =  pygame.transform.scale(pygame.image.load("./figs/foods/chicken.png"), (cell_size, cell_size))
pig =  pygame.transform.scale(pygame.image.load("./figs/foods/pig.png"), (cell_size, cell_size))
cattle =  pygame.transform.scale(pygame.image.load("./figs/foods/cattle.png"), (cell_size, cell_size))

food_images = [spinach, watermelon, strawberry, chicken, pig, cattle]
