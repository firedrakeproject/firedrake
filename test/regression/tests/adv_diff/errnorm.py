import pickle
import numpy as np


def errnorm(filename):
    with open(filename, "r") as f:
        a = pickle.load(f)

    return np.linalg.norm(a) / len(a)
