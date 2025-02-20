import itertools
import pickle
import random
import matplotlib.pyplot as plt

def generate_combinations(N_att, N_val, output_file="combinations.pkl"):
    # Create all possible combinations
    values = list(range(1, N_val + 1))
    combinations = list(itertools.product(values, repeat=N_att))
    combinations = [list(item) for item in combinations]
    # Shuffle combinations
    random.shuffle(combinations)
    
    # Split into training (90%) and test (10%) sets
    split_index = int(0.9 * len(combinations))
    train_combinations = combinations[:split_index]
    test_combinations = combinations[split_index:]
    
    # Save to pickle file
    with open(output_file, "wb") as file:
        pickle.dump({"train": train_combinations, "test": test_combinations}, file)
    
    print(f"Saved {len(train_combinations)} training and {len(test_combinations)} test combinations to {output_file}")

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
# generate_combinations(N_att=N_att, N_val=N_val, output_file=f"natt{N_att}_nval{N_val}.pkl")

load_and_visualize_pickle("natt2_nval10.pkl")
