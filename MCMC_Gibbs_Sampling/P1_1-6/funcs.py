import numpy as np

rng = np.random.default_rng()
gen_uniform = lambda a, b: rng.uniform(a, b)
gen_integer = lambda a, b: rng.integers(a, b)
gen_normal = lambda mu, std: rng.normal(mu, std)
gen_beta = lambda a, b: rng.beta(a, b)
gen_poiss = lambda l: rng.poisson(l)

# Обчислення елементів інваріантного розподілу
def get_stationary_dist(transition, iters=50):
    initial = np.zeros(transition.shape[0])
    initial[0] = 1

    for _ in range(iters):
        updated_initial = initial @ transition
        initial = updated_initial
        
    return initial.round(3)

# Обчислення матриці пропозицій 
def get_proposal_matrix(M):
    Q = np.zeros((M, M))
    Q[0, 0], Q[M-1, M-1] = 0.5, 0.5

    for i in range(M):
        if i == 0:
            Q[i, i+1] = 0.5
        elif i == M-1:
            Q[i, i-1] = 0.5
        else:
            Q[i, i-1], Q[i, i+1] = 0.5, 0.5

    return Q

# Рішення, чи приймати пропозицію
def proposal(alpha, i, j):
    v = gen_uniform(0, 1)
    
    if v <= alpha: return j
    elif v > alpha: return i
