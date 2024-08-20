import numpy as np

class OneHotEncoder:
    def __init__(self, n):
        self.n = n

    def __call__(self, label):
        x = np.zeros(self.n)
        x[label] = 1.0
        return x