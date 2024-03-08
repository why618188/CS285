from typing import Tuple
import numpy as np


class RandomAgent:
    def __init__(self, observation_shape: Tuple[int, ...], num_actions: int):
        super().__init__()
        self.num_actions = num_actions
        self.observation_shape = observation_shape

    def get_action(self, *args, **kwargs):
        # TODO(student): Return a random action
        return np.random.randint(self.num_actions)
    
    def update(self, *args, **kwargs):
        # Update is a no-op for the random agent
        return {}

# python cs285/scripts/run_hw5_explore.py -cfg experiments/exploration/pointmass_easy_random.yaml --dataset_dir datasets/
# python cs285/scripts/run_hw5_explore.py -cfg experiments/exploration/pointmass_medium_random.yaml --dataset_dir datasets/
# python cs285/scripts/run_hw5_explore.py -cfg experiments/exploration/pointmass_hard_random.yaml --dataset_dir datasets/
