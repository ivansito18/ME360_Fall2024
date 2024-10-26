import numpy as np
import matplotlib.pyplot as plt

N = 16

y = np.zeros(N)
u = np.ones(N-1)

y[0] = 0.

for n in range(N-1):
	y[n+1] = y[n] + u[n]


f1, ax = plt.subplots(1,1)
ax.stem(np.arange(N),y)
ax.set_ylabel("y")
ax.set_xticks(np.arange(N))
ax.set_xticklabels(np.arange(-1,N-1))

f1.suptitle("Problem 4 demo")
plt.show()
plt.close()