import funcs as f 
import numpy as np
import matplotlib.pyplot as plt
from MarkovChain import MarkovChain, find_distribution

'''
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
'''


# -------TASK 2---------
for k in range(3):
     choice = np.random.choice([0, 1])
     a = f.gen_uniform(0, 1) if choice else f.gen_integer(1, 5)
     a = np.round(a, 3)
     M = f.gen_integer(3, 10)

     Q = f.get_proposal_matrix(M)
     fig, axs = plt.subplots(2, 2, layout='constrained')
     for n in [100, 1000, 10000, 100000]:
          if n == 100: i, j = 0, 0
          elif n == 1000: i, j = 0, 1
          elif n == 10000: i, j = 1, 0
          elif n == 100000: i, j = 1, 1

          distribution = find_distribution(Q, N=n, type='zipf', a=a)

          def zipf(M, a):
               distribution = [0] * M
               denominator = sum([1 / (j ** a) for j in range(1, M+1)])
               for k in range(1, M+1):
                    distribution[k-1] = 1 / (k ** a) / denominator
               
               return distribution

          zipf = zipf(M, a)
          print('zipf: {}'.format(zipf))

          X = list(range(1, M+1))
          Y = distribution

          axs[i, j].plot(X, Y, color='black', label='Змодельований розподіл')
          axs[i, j].plot(X, zipf, color='pink', label='Розподіл Зіпфа')
          axs[i, j].scatter(X, Y)
          axs[i, j].scatter(X, zipf)
          axs[i, j].set_xlabel('Стан')
          axs[i, j].set_ylabel('Частота')
          title = 'M = {}, a = {}, N = {}'.format(M, a, n)
          axs[i, j].set_title(title, fontdict={'fontsize': 9})
          axs[i, j].legend()
     fig.savefig('task_2_{}.png'.format(k))