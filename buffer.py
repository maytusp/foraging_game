import sys
from typing import Dict, List, Tuple
import numpy as np
import collections
from collections import namedtuple, deque
import random

'''
Some parts are adopted from https://github.com/keep9oing/DRQN-Pytorch-CartPole-v1/
'''



class EpisodeData:
    """A simple numpy replay buffer."""

    def __init__(self):
        self.image = []
        self.loc = []
        self.action = []
        self.reward = []
        self.next_image = []
        self.next_loc = []
        self.done = []

    def put(self, transition):
        self.image.append(transition[0])
        self.loc.append(transition[1])
        self.action.append(transition[2])
        self.reward.append(transition[3])
        self.next_image.append(transition[4])
        self.next_loc.append(transition[5])
        self.done.append(transition[6])

    def sample(self, random_update=False, lookup_step=None, idx=None) -> Dict[str, np.ndarray]:
        image = np.array(self.image)
        loc = np.array(self.loc)
        action = np.array(self.action)
        reward = np.array(self.reward)
        next_image = np.array(self.next_image)
        next_loc = np.array(self.next_loc)
        done = np.array(self.done)

        if random_update is True:
            image = image[idx:idx+lookup_step]
            loc = loc[idx:idx+lookup_step]
            action = action[idx:idx+lookup_step]
            reward = reward[idx:idx+lookup_step]
            next_image = next_image[idx:idx+lookup_step]
            next_loc = next_loc[idx:idx+lookup_step]
            done = done[idx:idx+lookup_step]

        return dict(image=image,
                    loc=loc,
                    acts=action,
                    rews=reward,
                    next_image=next_image,
                    next_loc=next_loc,
                    done=done)

    def __len__(self) -> int:
        return len(self.image)


# Replay Buffer for Sequential Data
class EpisodeReplayBuffer:
    def __init__(self, max_epi_num, max_epi_len, lookup_step):
        self.buffer = deque(maxlen=max_epi_num)
        self.max_epi_len = max_epi_len
        self.lookup_step = lookup_step

    def add(self, episode_data):
        self.buffer.append(episode_data)

    def sample(self, batch_size):
        sampled_buffer = []

        ##################### RANDOM UPDATE ############################
        sampled_episodes = random.sample(self.buffer, batch_size)
        
        min_step = self.max_epi_len

        for episode in sampled_episodes:
            min_step = min(min_step, len(episode)) # get minimum step from sampled episodes
        
        # if min_step > 3:
        #     print("min step", min_step)

        for episode in sampled_episodes:
            if min_step > self.lookup_step: # sample buffer with lookup_step size
                idx = np.random.randint(0, len(episode)-self.lookup_step+1)
                sample = episode.sample(random_update=True, lookup_step=self.lookup_step, idx=idx)
                sampled_buffer.append(sample)
            else:
                idx = np.random.randint(0, len(episode)-min_step+1) # sample buffer with minstep size
                sample = episode.sample(random_update=True, lookup_step=min_step, idx=idx)
                sampled_buffer.append(sample)
 

        return sampled_buffer, len(sampled_buffer[0]) # buffers, sequence_length

    def __len__(self):
        return len(self.buffer)