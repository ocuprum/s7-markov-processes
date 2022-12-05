import numpy as np
from funcs import gen_uniform, gen_integer

in_interval = lambda u, a, b: 1 if a <= u < b else 0

class MarkovChain:
    def __init__(self, transition):
        self.transition = transition
        self.size = len(self.transition)
        self.time = 0
        self.x = gen_integer(0, self.size)
        self.__get_distances()
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
    
    def __psi(self, u):
        d = self.D[self.x]

        result = 0
        for j in range(self.size):
            if d[j] == 0: break
            d1, d2 = d[j]
            if d1 == d2: continue
            result += in_interval(u, d1, d2) * j
        
        return result
        
    def next(self, type: str):
        if type == 'psi':
            u = gen_uniform(0, 1)
            self.x = self.__psi(u)
        
        self.time += 1
        return self.x

# Пошук частот, наближених до інваріантного розподілу
def find_distribution(mtrx, N, type, start=1000):
    mchain = MarkovChain(mtrx)
    distribution = [0] * mchain.size
    for n in range(N + start):
        x = mchain.next(type)
        if n <= start: continue
        distribution[x] += 1
        if n in np.array([50, 250, 500, 1000, 10000, 99999]) + start:
            print('N = {}, Результат: {}'.format(n - start, np.array([d / (n-start) for d in distribution]).round(3)))
    
    distribution = [d / N for d in distribution]

    return distribution