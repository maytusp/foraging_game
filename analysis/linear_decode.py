import pickle
import numpy as np
import sklearn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    log_file_path = "../logs/goal_condition_pickup/dec_ppo_invisible_possig/grid5_img5_ni2_natt2_nval10_nw16_998400000/seed1/mode_train/normal/trajectory.pkl"
    num_episodes = 2000