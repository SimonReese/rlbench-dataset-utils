from typing import List

import numpy as np

from .observation import Observation


class Demo(object):
    """ A Demo is an instance of complete execution of a task, that is a specific episode for a specific variation for a specific task.

    Contains a list of observations, where each observation is a step in an instant
    """

    def __init__(self, observations: List[Observation], random_seed=None, num_reset_attempts=None, demo_description = None):
        self._observations = observations
        self.random_seed = random_seed
        self.num_reset_attempts = num_reset_attempts
        self.demo_description = demo_description
    
    def __len__(self):
        return len(self._observations)

    def __getitem__(self, i) -> Observation:
        return self._observations[i]
    
    def __iter__(self):
        return iter(self._observations)

    def restore_state(self):
        np.random.set_state(self.random_seed) # type: ignore