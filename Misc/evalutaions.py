import numpy as np


# Function to get the optimal policy given a q-table.
def get_optimal_policy(Q):
    return np.argmax(Q, axis=-1)