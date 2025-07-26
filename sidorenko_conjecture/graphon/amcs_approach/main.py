import numpy as np
import itertools
import random
from itertools import product
from time import time
from helpers import *
from amcs import *

if __name__ == "__main__":
    ''' ---- Define the graph H ---- '''
    H = np.array([
        [0,0,0,0,0,0,0,1,1,1], [0,0,0,0,0,1,0,0,1,1], [0,0,0,0,0,1,1,0,0,1],
        [0,0,0,0,0,1,1,1,0,0], [0,0,0,0,0,0,1,1,1,0], [0,1,1,1,0,0,0,0,0,0],
        [0,0,1,1,1,0,0,0,0,0], [1,0,0,1,1,0,0,0,0,0], [1,1,0,0,1,0,0,0,0,0],
        [1,1,1,0,0,0,0,0,0,0]
    ])

    ''' ---- Define initial W ---- '''
    '''IMPORTANT: W MUST BE SYMMETRIC'''
    W_initial = np.array([
        [0.2, 0.3, 0.1, 0.541, 0.44],
        [0.3, 0.534, 0.395, 0.497, 0.573],
        [0.1, 0.395, 0.527, 0.467, 0.475],
        [0.541, 0.497, 0.467, 0.53, 0.425],
        [0.44, 0.573, 0.475, 0.425, 0.579]
    ])
    # W_initial = np.random.rand(5, 5)
    # W_initial = (W_initial + W_initial.T) / 2


    # Run the AMCS Optimization
    start_time = time()
    W_final, final_gap = AMCS_graphon(H, W_initial, max_depth=5, max_level=3)
    print(f"\nTotal search time: {time() - start_time:.2f} seconds")
