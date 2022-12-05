import funcs as f 
import numpy as np
from MarkovChain import MarkovChain, find_distribution

# -------TASK 1---------
N = 1000

P = [[0.5, 0.5, 0, 0, 0, 0],
     [0.5, 0, 0.5, 0, 0, 0],
     [0, 0, 0.5, 0.3, 0.2, 0],
     [0, 0, 0.3, 0, 0, 0.7],
     [0, 0, 0.2, 0, 0.8, 0],
     [0, 0, 0, 0.7, 0, 0.3]]

theoretic = f.get_stationary_dist(np.matrix(P), 10000)
print('Інваріантний розподіл матриці P:{}'.format(theoretic))

for _ in range(3):
     find_distribution(P, N=100000, type='psi')
     print()