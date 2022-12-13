import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm

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

print(np.mean(X), np.var(X))
print(np.mean(Y), np.var(Y))
fig, axs = plt.subplots(2, 1, layout='constrained')
axs[0].hist(X, 100, width=0.2, color='pink')
axs[0].set_title('Умовний розподіл x_(n+1) | y_n', fontdict={'size': 10})
axs[1].hist(Y, 100, width=0.2, color='violet')
axs[1].set_title('Умовний розподіл y_(n+1) | x_(n+1)', fontdict={'size': 10})
fig.savefig('task_7.png')


