import funcs as f 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from MarkovChain import MarkovChain, find_distribution
from ColorGraph import ColorGraph, get_colorings

print('\n-------------------TASK 1-------------------')
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


print('\n-------------------TASK 2-------------------')
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
          print('M = {}, a = {}, N = {}'.format(M, a, n))
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


print('\n-------------------TASK 3-------------------')
print('result -> png')
# -------TASK 3---------
for gr in range(3):
     if gr == 0: a, b = (0.5, 0.5)
     if gr == 1: a, b = (6, 8)
     if gr == 2: a, b = (4, 2)

     for parts in [25, 100, 1000]:

          fig, axs = plt.subplots(2, 2, layout='constrained')
          for n in [100, 1000, 10000]:
               if n == 100: i, j = 0, 0
               elif n == 1000: i, j = 0, 1
               elif n == 10000: i, j = 1, 0

               distribution = find_distribution(N=n, type='beta', a=a, b=b, parts=parts)

               X, Y = [], []
               for key, value in sorted(distribution.items()):
                    X.append(key)
                    Y.append(value)

               rv = beta(a, b)
               w = X[1] - X[0]

               axs[i, j].set_ylim([0, max(Y)])
               axs[i, j].bar(X, Y, width=w, color='violet', label='Змодельований розподіл')
               title = 'a = {}, b = {}, parts = {}, N = {}'.format(a, b, parts, n)
               axs[i, j].set_title(title, fontdict={'fontsize': 9})
               axs[i, j].legend()
          i, j = 1, 1
          x = np.linspace(0, 1)
          axs[i, j].plot(x, rv.pdf(x), color='black', label='Beta-розподіл')
          title = 'a = {}, b = {}'.format(a, b)
          axs[i, j].set_title(title, fontdict={'fontsize': 9})
          axs[i, j].legend()
          fig.savefig('task_3_prts_{}_n_{}_a_{}_b_{}.png'.format(parts, n, a, b))

print('\n-------------------TASK 4-------------------')
print('result -> png')
# -------TASK 4---------
y = 3
mu = 0
sigma = 1
tau = 2
d = 1
k = 10000
parts = 1000


fig1, axs1 = plt.subplots(3, 1)
for d in [0.1, 1, 100]:
     if d == 0.1: 
          g = 0 
          color = 'blue'
     elif d == 1: 
          g = 1
          color = 'pink'
     elif d == 100: 
          g = 2
          color = 'violet'

     distribution, chain = find_distribution(N=k, type='apost', d=d, y=y, mu=mu, sigma=sigma, tau=tau, parts=parts)

     X = range(len(chain))
     axs1[g].plot(X, chain, color=color, linewidth=0.4, label='d = {}'.format(d))
     axs1[g].set_xlabel('n')
     axs1[g].set_ylabel('X_n')
     axs1[g].legend(loc='best', prop={'size': 8})

     if d == 1:
          fig, ax = plt.subplots()
          X, Y = [], []
          for key, value in sorted(distribution.items()):
               X.append(key)
               Y.append(value)

          w = X[1] - X[0]

          tmu, tsigma = 2.4, 0.8
          mu, sigma = np.mean(chain).round(4), np.std(chain)

          ax.set_ylim([0, 0.6])
          count, bins, ignored = plt.hist(chain, 100, density=True, color='grey', alpha=0.4, label='Змодельований розподіл')
          ax.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                    np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
          linewidth=2, color='blue', label='Змодельований розподіл: mu = {}, sigma^2 = {}'.format(mu, np.var(chain).round(4)))
          ax.plot(bins, 1/(tsigma * np.sqrt(2 * np.pi)) *
                    np.exp( - (bins - tmu)**2 / (2 * tsigma**2) ),
          linewidth=2, color='violet', label='Теоретичний розподіл: mu = {}, sigma^2 = {}'.format(tmu, tsigma))
          
          title = 'k = {}'.format(k)
          ax.set_title(title)
          ax.legend(loc='best', prop={'size': 6})
          fig.savefig('task_4_k_{}.png'.format(parts, k))
fig1.savefig('task_4_dif_d')


print('\n-------------------TASK 5-------------------')
# -------TASK 5---------
N = 10000

# k = 3
V = range(10)
V_color = ['green', 'blue', 'yellow', 'blue', 'yellow', 
           'green', 'yellow', 'green', 'blue', 'blue']
colors = ['green', 'blue', 'yellow']
E = [[1], [0, 2, 5], [1, 3], [2, 4], [3, 5, 8, 9], 
     [1, 4, 6, 8], [5], [8], [4, 5, 7], [4]]

cg = ColorGraph(V, V_color, colors, E)

C, frequency = get_colorings(cg, N)
print("Кількість різних розфарбовок при кількості кольорів {}: {}".format(len(colors), C))

fig, axs = plt.subplots(2, 1)
axs[0].plot(range(len(frequency)), frequency, color='pink', label='k = {}'.format(len(colors)))

# k = 4
V_color = ['pink', 'yellow', 'pink', 'blue', 'yellow', 
           'green', 'pink', 'green', 'blue', 'pink']
colors = ['green', 'blue', 'yellow', 'pink']

cg = ColorGraph(V, V_color, colors, E)

C, frequency = get_colorings(cg, N)
print("Кількість різних розфарбовок при кількості кольорів {}: {}".format(len(colors), C))
axs[1].plot(range(len(frequency)), frequency, color='violet', linewidth=0.3, label='k = {}'.format(len(colors)))

axs[0].set_title('Частоти виникнень різних станів-розфарбовок\n(стани пронумеровані у порядку їх виникнення при генеруванні нових розфарбовок)', fontdict={'fontsize': 9})
axs[0].legend(loc='best', prop={'size': 6})
axs[1].legend(loc='best', prop={'size': 6})
fig.savefig('task_5'.format(len(colors)))


print('\n-------------------TASK 6-------------------')
# -------TASK 6---------

l = 10
a, b = 1, 1
x = 7
k = 10000

n = f.gen_integer(x, x + 10)

P, N, P_apost, N_apost = [], [], {}, {}
for _ in range(k):
     p = np.round(f.gen_beta(x + a, n - x + b), 2)
     y = f.gen_poiss(l * (1 - p))
     n = x + y

     P.append(p)
     P_apost[p] = P_apost.get(p, 0) + 1
     N_apost[n] = N_apost.get(n, 0) + 1

P_apost = {p: value / k for p, value in sorted(P_apost.items())}
N_apost = {n: value / k for n, value in sorted(N_apost.items())}

fig, axs = plt.subplots(2, 1, layout='constrained')
plt.xticks(range(x-1, max(N_apost.keys()), 1))
axs[0].bar(P_apost.keys(), P_apost.values(), width=0.01, color='pink')
axs[0].set_title('Апостеріорний розподіл p', fontdict={'fontsize': 10})
axs[1].bar(N_apost.keys(), N_apost.values(), color='violet')
axs[1].set_title('Апостеріорний розподіл N', fontdict={'fontsize': 10})
fig.savefig('task_6.png')

print('Оцінка p як вибіркового середнього отриманої вибірки: {}'.format(np.mean(P)))