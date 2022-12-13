import numpy as np 
import matplotlib.pyplot as plt
from ContingencyTable import ContingencyTable, tbl_to_tuple

'''
rng = np.random.default_rng()

print('\n-------------------TASK 7-------------------')
x, y = 1, 1
ro = 0.5
N = 10000

X, Y = [], []

for _ in range(N):
    x = rng.normal(ro * y, np.sqrt(1 - ro ** 2))
    y = rng.normal(ro * x, np.sqrt(1 - ro ** 2))
    X.append(x)
    Y.append(y)

print('M(x_(n+1) | y_n) = {}, D(x_(n+1) | y_n) = {}'.format(np.mean(X), np.var(X)))
print('M(y_(n+1) | x_(n+1)) = {}, D(y_(n+1) | x_(n+1)) = {}'.format(np.mean(Y), np.var(Y)))
fig, axs = plt.subplots(2, 1, layout='constrained')
axs[0].hist(X, 100, width=0.2, color='pink')
axs[0].set_title('Умовний розподіл x_(n+1) | y_n', fontdict={'size': 10})
axs[1].hist(Y, 100, width=0.2, color='violet')
axs[1].set_title('Умовний розподіл y_(n+1) | x_(n+1)', fontdict={'size': 10})
fig.savefig('task_7.png')
'''

print('\n-------------------TASK 8-------------------')
            
N = 10000
print('Кількість ітерацій - {}'.format(N))
r = (3, 2, 1)
c = (2, 2, 1, 1)
table = [[1, 1, 0, 1],
         [1, 1, 0, 0],
         [0, 0, 1, 0]]

# not MCMC 
ct = ContingencyTable(table, r, c)

cts_not_MCMC = {}
for _ in range(N):
    tpl = tbl_to_tuple(ct.next_not_MCMC())
    cts_not_MCMC[tpl] = cts_not_MCMC.get(tpl, 0) + 1

def addlabels(x,y, a):
    for i in range(len(x)):
        a.text(i,y[i],y[i], ha='center', size=9)

fig, axs = plt.subplots(2, 1, layout='constrained')
axs[0].bar(range(len(cts_not_MCMC.keys())), [v / N for v in cts_not_MCMC.values()], color='pink')
axs[0].set_title('Not MCMC, кількість ітерацій - {}\n'.format(N), fontdict={'size': 10})
axs[0].set_ylim(0, 0.15)
addlabels(range(len(cts_not_MCMC.keys())), [v / N for v in cts_not_MCMC.values()], axs[0])
print('Метод not MCMC: інваріантний розподіл = {}'.format([v / N for v in cts_not_MCMC.values()]))

# MCMC
ct = ContingencyTable(table, r, c)

cts_MCMC = {}
for _ in range(N):
    tpl = tbl_to_tuple(ct.next_MCMC())
    cts_MCMC[tpl] = cts_MCMC.get(tpl, 0) + 1

axs[1].bar(range(len(cts_MCMC.keys())), [v / N for v in cts_MCMC.values()], color='violet')
axs[1].set_title('MCMC, кількість ітерацій - {}\n'.format(N), fontdict={'size': 10})
axs[1].set_ylim(0, 0.18)
addlabels(range(len(cts_MCMC.keys())), [v / N for v in cts_MCMC.values()], axs[1])
print('Метод MCMC: інваріантний розподіл = {}'.format([v / N for v in cts_MCMC.values()]))
fig.savefig('task_8.png')