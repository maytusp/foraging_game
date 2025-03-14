import random

possible_networks = [i for i in range(3)]
possible_pairs = [[i,(i+1) % 3] for i in range(3)]

for i in range(20):
    selected_networks = random.sample(possible_pairs, 1)[0]
    print(selected_networks)
