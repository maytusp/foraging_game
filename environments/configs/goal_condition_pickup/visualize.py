import itertools
import pickle
import random
import matplotlib.pyplot as plt

def load_and_visualize_pickle(input_file="combinations.pkl"):
    with open(input_file, "rb") as file:
        data = pickle.load(file)
    
    train_combinations = data["train"]
    test_combinations = data["test"]
    print(test_combinations)
    plt.figure(figsize=(8, 6))
    train_x, train_y = zip(*train_combinations)
    test_x, test_y = zip(*test_combinations)
    
    plt.scatter(train_x, train_y, color='blue', label='Train', alpha=0.5)
    plt.scatter(test_x, test_y, color='red', label='Test', alpha=0.5)
    plt.xlabel("Attribute 1")
    plt.ylabel("Attribute 2")
    plt.legend()
    plt.title("Visualization of Training and Test Combinations")
    plt.show()

# Example usage
N_att = 2
N_val = 10

load_and_visualize_pickle(f"natt{N_att}_nval{N_val}.pkl")
