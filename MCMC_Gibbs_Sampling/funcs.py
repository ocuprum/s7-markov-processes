import numpy as np

rng = np.random.default_rng()
gen_uniform = lambda a, b: rng.uniform(a, b)
gen_integer = lambda a, b: rng.integers(a, b)

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
def proposal(i, j, a, v):

    alpha = min((i ** a) / (j ** a), 1)
    v = np.random.default_rng().uniform(0, 1)
    
    if v <= alpha:
        return j
    elif v > alpha:
        return i

# Генерація початкового розподілу ланцюга Маркова
def initial_distribution(N, epsilon=0.01):
    start = 1 / N
    distribution = [0] * N

    for i in range(N-1):
        distribution[i] = round(gen_uniform(start - epsilon, start), 3)
    distribution[-1] = 1 - sum(distribution[:-1])

    return distribution