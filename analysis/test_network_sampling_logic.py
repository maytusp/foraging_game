import numpy as np
possible_networks = [0,1,2]
num_agents = 2
for i in range(10):
    selected_networks = np.random.choice(possible_networks, num_agents, replace=False)
    print(selected_networks)