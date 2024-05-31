import numpy as np


def random_3d_u():
    a = np.random.uniform(0, 2 * np.pi)
    z = np.random.uniform(-1, 1)
    return np.array([np.cos(a), np.sin(a), z])

