import numpy as np

class random_search_policy:

    def __init__(self):
        self.params = np.random.rand(4) * 2 - 1

    def action(self,observation):
        return 0 if np.matmul(self.params,observation) < 0 else 1
    