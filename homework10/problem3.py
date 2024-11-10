import numpy as np 
import matplotlib.pyplot as plt

T = 30.
Delta = 0.2
ITER = int(T/Delta)
a = .1



y = np.zeros(ITER)

u = np.sin(Delta*np.arange(ITER-1))

for iters in range(ITER-1):
	y[iters+1] = y[iters] + (-a*y[iters] + u[iters])*Delta


f1,axs = plt.subplots(1,1)
axs.plot(np.arange(0,ITER)*Delta, y)
axs.set_xlabel("Time(s)")
axs.set_ylabel("Solution of ODE")
plt.show()
plt.close()