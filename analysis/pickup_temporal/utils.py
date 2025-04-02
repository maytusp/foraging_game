import numpy as np
import pickle
def load_trajectory(file_path):
    with open(file_path, "rb") as f:
        log_data = pickle.load(f)
    return log_data
# check what item id the agent see
def get_food_id(agent_loc, food_loc_list):
    dist_list = []
    for food_loc in food_loc_list:
        food_loc = np.array(food_loc)
        curr_dist = np.linalg.norm(food_loc - agent_loc)
        dist_list.append(curr_dist)
        # print(curr_dist)
    return np.argmin(np.array(dist_list))

def check_comm_range(agent0_pos, agent1_pos, comm_range=1):
    '''
    check whether agents can communicate or not
    '''
    if (agent0_pos[0] >= agent1_pos[0] - comm_range and agent0_pos[0] <= agent1_pos[0] + comm_range and
        agent0_pos[1] >= agent1_pos[1] - comm_range and agent0_pos[1] <= agent1_pos[1] + comm_range):
        return 1
    else:
        return 0

def get_message_indices(agent0_pos_list, agent1_pos_list, episode_length):
    '''
    get message indices where agents are nearby
    '''
    T = len(agent0_pos_list)
    time_indices = []
    for t in range(episode_length):
        if check_comm_range(agent0_pos_list[t], agent1_pos_list[t]):
            time_indices.append(t)
    return time_indices
