import numpy as np
from funcs import gen_uniform, gen_integer, proposal

in_interval = lambda u, a, b: 1 if a <= u < b else 0
interval_part = lambda N, a, b: (b - a) / N

class MarkovChain:
    def __init__(self, transition, beta=None):
        self.transition = transition
        self.size = len(self.transition) if transition is not None else None
        self.time = 0
        self.x = gen_uniform(0, 1) if beta else gen_integer(0, self.size)
        if transition is not None: self.__get_distances()
        print('x_start = {}'.format(self.x))

    def __get_distances(self):
        self.D = [[0 for _ in range(self.size)] for _ in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                if j == 0:
                    self.D[i][j] = (0, self.transition[i][j])
                elif j != 0:
                    self.D[i][j] = (self.D[i][j-1][1], self.D[i][j-1][1] + self.transition[i][j])

                if self.D[i][j][1] == 1: break
    
    def __psi(self):
        u = gen_uniform(0, 1)
        d = self.D[self.x]

        result = 0
        for j in range(self.size):
            if d[j] == 0: break
            d1, d2 = d[j]
            if d1 == d2: continue
            result += in_interval(u, d1, d2) * j
        
        return result
    
    def __get_transition_matrix(self):
        self.proposal = self.transition.copy()
        self.transition = np.array([[0 for _ in range(self.size)] for _ in range(self.size)])

        for i in range(self.size):
            for j in range(self.size):
                if i == j:
                    self.transition[i, j] = 1 - sum([self.proposal[i, k] * self.alphas[k, j] for k in range(self.size) if k != i])
                else:
                    self.transition[i, j] = self.proposal[i, j] * self.alphas[i, j]

    def next(self, type: str, a=None, b=None):
        if type == 'psi':
            self.x = self.__psi()
        elif type == 'zipf':
            self.alphas = np.array([[0 for _ in range(self.size)] for _ in range(self.size)])
            i = self.x
            j = self.__psi()

            if i == j:
                self.x = i
            else:
                alpha = min([((i + 1) ** a) / ((j + 1) ** a), 1])
                self.alphas[i][j] = alpha
                self.x = proposal(alpha, i, j)

            self.__get_transition_matrix()
        elif type == 'beta':
            u = gen_uniform(0, 1)
            fy = lambda y, a, b: (y ** (a - 1)) * ((1 - y) ** (b - 1))
            fu_div_fx =  fy(u, a, b) / fy(self.x, a, b)
            alpha = min([fu_div_fx, 1])
            self.x = proposal(alpha, self.x, u)

        self.time += 1
        return self.x

# Пошук частот, наближених до інваріантного розподілу
def find_distribution(mtrx=None, N=1000, type=None, start=1000, a=None, b=None, parts=None):
    if type == 'psi' or type == 'zipf':
        mchain = MarkovChain(mtrx)
        distribution = [0] * mchain.size

        for n in range(N + start):
            x = mchain.next(type, a)
            if n <= start: continue
            distribution[x] += 1
            if n in np.array([50, 250, 500, 1000, 10000, 100000]) + start:
                print('N = {}, Результат: {}'.format(n - start, np.array([d / (n-start) for d in distribution]).round(3)))

        distribution = [d / N for d in distribution]

    elif type == 'beta':
        mchain = MarkovChain(transition=None, beta=True)
        distribution = {}

        part = interval_part(parts, 0, 1)
        for i in range(N + start):
            x = mchain.next(type, a=a, b=b)
            if i <= start: continue
            for j in range(parts):
                start = j * part
                if in_interval(x, start, start + part):
                    distribution[start] = distribution.get(start, 0) + 1
                    break

        distribution = {k: v / N for k, v in distribution.items()}
    
    return distribution