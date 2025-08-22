import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import editdistance
from language_analysis import Disent, TopographicSimilarity
import os
from collections import deque
# Load the .pkl file
def load_trajectory(file_path):
    with open(file_path, "rb") as f:
        log_data = pickle.load(f)
    return log_data

# Extract and prepare data for t-SNE
def extract_data(log_data):
    message_data = {"agent0": [], "agent1":[]}
    attribute_data = []
    scores = {"agent0": [], "agent1":[]}
    item_locs = {"agent0": [], "agent1":[]}
    for id, (episode, data) in enumerate(log_data.items()):

        log_s_messages = data["log_s_messages"]
        who_see_target = data["who_see_target"]
        target_score = data["log_target_food_dict"]["score"]
        target_loc = data["log_target_food_dict"]["location"] # (2,)
        distractor_score = data["log_distractor_food_dict"]["score"][0]
        distractor_loc = data["log_distractor_food_dict"]["location"][0] # (2,)

        for agent_id in range(2):
            messages = log_s_messages[:, agent_id].flatten()
            message_data[f"agent{agent_id}"].append(messages)  # Collect all time steps for the agent
            extract_attribute = [target_score, target_loc[0], target_loc[1], 
                                distractor_score, distractor_loc[0], distractor_loc[1]]
        attribute_data.append(extract_attribute)
    return message_data, attribute_data


def get_topsim(message_data,attribute_data, num_networks):
    sender_list = [i for i in range(num_networks)]
    data = []
    n_samples = 1000000
    extracted_message = []
    extracted_attribute = []
    receiver = 0
    avg_topsim = 0
    max_eval_episodes=1000
    max_message_length=5
    for sender in sender_list:
        extracted_message.append(np.array(message_data[f"{sender}-{receiver}"]["agent0"]))
        extracted_attribute.append(attribute_data[f"{sender}-{receiver}"])
        n_samples = min(extracted_message[sender].shape[0], n_samples)


    for agent_id in range(len(sender_list)):
        messages = np.array(extracted_message[sender_list[agent_id]])
        attributes = np.array(extracted_attribute[sender_list[agent_id]])
        topsim = TopographicSimilarity.compute_topsim(attributes[:max_eval_episodes], messages[:max_eval_episodes, :max_message_length])     
        avg_topsim += topsim
        print(f"agent_id {agent_id} has topsim {topsim}")
    avg_topsim /= num_networks

    return avg_topsim

def get_similarity(message_data, num_networks, CONNECTIVITY):
    sender_list = [i for i in range(num_networks)]
    data = []
    similarity_mat = np.zeros((num_networks,num_networks))
    n_samples = 1000000
    extracted_message = []

    for sender in sender_list:
        # Find a trained receiver for a specific sender
        # This requires connectivity pairs
        for connected_nodes in CONNECTIVITY:
            node1, node2 = connected_nodes.split("-")
            if sender == int(node1):
                receiver = int(node2)
                break
        # print(f"{receiver}-{sender}")
        extracted_message.append(np.array(message_data[f"{receiver}-{sender}"]["agent0"]))
        n_samples = min(extracted_message[sender].shape[0], n_samples)


    for first_agent_id in range(len(sender_list)):
        for second_agent_id in range(len(sender_list)):
            for i in range(n_samples):          
                m1 = extracted_message[sender_list[first_agent_id]][i]
                m2 = extracted_message[sender_list[second_agent_id]][i]

                m1 = [i for i in m1 if i != -1]
                m2 = [i for i in m2 if i != -1]

                dist = editdistance.eval(m1, m2) / max(len(m1), len(m2))
               
                similarity_mat[first_agent_id,second_agent_id] += (1 - dist)

    similarity_mat = similarity_mat / n_samples
    mask = np.ones_like(similarity_mat)
    mask[np.triu_indices_from(mask, k=0)] = 0
    avg_sim = np.sum(similarity_mat * mask) / np.sum(mask)

    return similarity_mat , avg_sim


def plot_heatmap(similarity_mat, saved_fig_path):
    # mask = np.zeros_like(similarity_mat)
    # mask[np.triu_indices_from(mask)] = True
    mask = np.triu(np.ones_like(similarity_mat, dtype=bool), k=1)  # Mask only upper triangle (excluding diagonal)
    with sns.axes_style("white"):
        ax = sns.heatmap(similarity_mat, mask=mask, square=True,  cmap="YlGnBu")
        plt.savefig(saved_fig_path)
        plt.close()

def load_score(filename):
    scores = {}
    with open(filename, "r") as f:
        for line in f:
            x = line.strip().split(": ")
            if "Reward" in x[0]:
                continue
            scores[x[0].strip()] = float(x[1].strip())
    return scores

def compute_all_pairs_shortest_paths(num_nodes, edges_string):
    """
    Compute shortest path lengths (social distances) between all pairs of nodes.
    Returns a matrix distances[i][j] = shortest path length from i to j.
    """
    edges = []
    for pair in edges_string:
        i,j = pair.split("-")
        i,j = int(i), int(j)
        edges.append([i,j])
    # Build adjacency list
    graph = {i: [] for i in range(num_nodes)}
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)  # assuming undirected graph

    # Initialize distance matrix
    distances = [[float('inf')] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        distances[i][i] = 0

    # BFS for each node to compute shortest paths
    for start in range(num_nodes):
        queue = deque([start])
        visited = {start}
        while queue:
            current = queue.popleft()
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[start][neighbor] = distances[start][current] + 1
                    queue.append(neighbor)

    return np.array(distances)

def average_score_by_distance(score_mat, dis_mat):
    """
    Compute average score (ls or sr) for each social distance.

    Args:
        score_mat (np.ndarray): NxN matrix of scores (sucess rate or language similarity).
        dis_mat (np.ndarray): NxN matrix of distances between agents.

    Returns:
        np.ndarray: 1D array where element i is the average score for pairs at distance i+1.
    """
    max_distance = np.max(dis_mat)


    num_seeds = score_mat.shape[0]
    scores = {d:[] for d in range(1, max_distance + 1)}
    avg_scores = np.zeros((num_seeds, max_distance)) # avg_score across pairs with the same distance within one seed
    for d in range(1, max_distance + 1):
        for s in range(num_seeds):
            # Find all pairs at distance d (excluding self-pairs)
            idx = np.where(dis_mat == d)
            pairs = [(i, j) for i, j in zip(*idx) if i < j]
            if pairs:
                sims = [score_mat[s, i, j] for i, j in pairs]
                scores[d] += sims
            else:
                scores[d] = np.nan  # No pairs at this distance
            avg_scores[s, d-1] = np.mean(sims) # array starts from idx 0 to D-1

    final_mean = np.mean(avg_scores, axis=0)
    final_std = np.std(avg_scores, axis=0)


    return list(final_mean), list(final_std)

    

if __name__ == "__main__":
    checkpoints_dict = {
        "wsk4p02_ppo_15net_invisible": {'seed1': 665600000, 'seed2': 819200000, 'seed3':819200000},
        "optk2e30_ppo_15net_invisible": {'seed1': 665600000, 'seed2': 819200000, 'seed3':819200000},
        "ccnet_ppo_15net_invisible" : {'seed1': 1126400000, 'seed2': 819200000, 'seed3':819200000},
        }
    connected_dict = {"wsk4p02_ppo_15net_invisible": [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [3, 5], [4, 5], [4, 6], [5, 7], [6, 7], [7, 8], [8, 9], [8, 10], [9, 10], [10, 12], [11, 12], [11, 13], [12, 13], [14, 0], [14, 1], [5, 13], [6, 7], [7, 10], [9, 6], [10, 8], [12, 5], [13, 6], [13, 5]],
                     "optk2e30_ppo_15net_invisible": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 0], [0, 7], [0, 8], [1, 8], [1, 9], [2, 9], [2, 10], [3, 10], [3, 11], [4, 11], [4, 12], [5, 12], [5, 13], [6, 13], [6, 14], [7, 14]],
                     "ccnet_ppo_15net_invisible": [(3, 4), (12, 13), (0, 2), 
                                                    (8, 9), (9, 11), (0, 14), (13, 14), 
                                                    (6, 8), (4, 5), (5, 6), (0, 1), (9, 10), 
                                                    (1, 2), (10, 11), (6, 7), (3, 5), 
                                                    (12, 14), (2, 3), (11, 12), (7, 8)],
                    }
    mean_std_by_model = {model_name : {"mean_ls":[], "std_ls":[], "mean_sr":[], "std_sr":[]} for model_name in checkpoints_dict.keys()}
    SEEDS = [1, 2,3]
    num_seed = len(SEEDS)
    for model_name in checkpoints_dict.keys():
        print(f"MODEL NAME: {model_name}")
        # This is to be changed based on the connectivity metric of small world network
        pairs = [f"{pair[0]}-{pair[1]}" for pair in connected_dict[model_name]]
        
        # Add swapped pairs
        all_pairs = set(pairs)  # use a set to avoid duplicates
        for p in pairs:
            a, b = p.split('-')
            swapped = f"{b}-{a}"
            all_pairs.add(swapped)

        # Convert back to list if needed
        CONNECTIVITY = all_pairs
        for num_networks in [15]:
            all_seed_ls_mat = np.zeros((num_seed, num_networks,num_networks))
            all_seed_sr_mat = np.zeros((num_seed, num_networks,num_networks))
            for s_idx, seed in enumerate(SEEDS):
                
                ckpt_name = checkpoints_dict[model_name][f"seed{seed}"]
                combination_name = f"grid5_img3_ni2_nw4_ms10_{ckpt_name}"

                print(f"{model_name}/{combination_name}")
                saved_fig_dir = f"plots/population/small_world/sr_lang_sim"
                saved_score_dir = f"../../logs/small_world_pop/exp2/{model_name}/{combination_name}_seed{seed}"
                saved_fig_path_langsim = os.path.join(saved_fig_dir, f"{model_name}_{combination_name}_similarity.png")
                saved_fig_path_sr = os.path.join(saved_fig_dir, f"{model_name}_{combination_name}_sr.png")
                os.makedirs(saved_fig_dir, exist_ok=True)
                os.makedirs(saved_score_dir, exist_ok=True)
                mode = "test"
                network_pairs = [f"{i}-{j}" for i in range(num_networks) for j in range(num_networks)]
                log_file_path = {}
                sr_dict = {}
                sr_mat = np.zeros((num_networks, num_networks))
                message_data = {}
                attribute_data = {}
                
                # For Interchangeability
                ic_numerator = []
                ic_denominator = []
                
                for pair in network_pairs:
                    row, col = pair.split("-")
                    row, col = int(row), int(col)
                    # print(f"loading network pair {pair}")
                    log_file_path[pair] =  f"../../logs/population/pickup_high_v1/{model_name}/{pair}/{combination_name}/seed{seed}/mode_{mode}/normal/trajectory.pkl"
                    sr_dict[pair] = load_score(f"../../logs/population/pickup_high_v1/{model_name}/{pair}/{combination_name}/seed{seed}/mode_{mode}/normal/score.txt")
                    sr_mat[row, col] = sr_dict[pair]["Success Rate"]
                    if row == col:
                        ic_numerator.append(sr_dict[pair]["Success Rate"])
                    elif min(abs(row-col), num_networks-abs(row-col)) == 1:
                        ic_denominator.append(sr_dict[pair]["Success Rate"])
                    # Load log data
                    log_data = load_trajectory(log_file_path[pair])

                    # Prepare data for t-SNE
                    message_data[pair], attribute_data[pair] = extract_data(log_data)
                    

                ic = np.mean(ic_numerator) / np.mean(ic_denominator)
                ls_mat, avg_sim = get_similarity(message_data, num_networks, CONNECTIVITY)
                
                # Save the variables
                np.savez(os.path.join(saved_score_dir, "sim_scores.npz"), 
                                        ls_mat=ls_mat, 
                                        avg_sim=avg_sim, 
                                        sr_mat=sr_mat,
                                        ic=ic)


                all_seed_ls_mat[s_idx, :, :] = ls_mat
                all_seed_sr_mat[s_idx, :, :] = sr_mat


            distance_mat = compute_all_pairs_shortest_paths(15, CONNECTIVITY)
            avg_sr_mat = np.mean(all_seed_sr_mat, axis=0)
            avg_ls_mat = np.mean(all_seed_ls_mat, axis=0)
            # This is ready to be plotted
            avg_ls_list, std_ls_list = average_score_by_distance(all_seed_ls_mat, distance_mat)
            avg_sr_list, std_sr_list = average_score_by_distance(all_seed_sr_mat, distance_mat)
            print(f"LS: {avg_ls_list} +- {std_ls_list}")
            print(f"SR: {avg_sr_list} +- {std_sr_list}")
            mean_std_by_model[model_name]["mean_ls"] = avg_ls_list
            mean_std_by_model[model_name]["std_ls"] = std_ls_list
            mean_std_by_model[model_name]["mean_sr"] = avg_sr_list
            mean_std_by_model[model_name]["std_sr"] = std_sr_list

            np.savez(os.path.join(saved_score_dir, "sr_ls_dist.npz"), 
                                                avg_ls_mat=avg_ls_mat, 
                                                avg_sr_mat=avg_sr_mat,
                                                distance_mat=distance_mat,)
            plot_heatmap(avg_ls_mat, saved_fig_path_langsim)
            plot_heatmap(avg_sr_mat, saved_fig_path_sr)

    with open(os.path.join(saved_fig_dir, "mean_std_by_model.pkl"), 'wb') as file:
        pickle.dump(mean_std_by_model, file)