import numpy as np 
import matplotlib.pyplot as plt

T = 30.
Delta = 0.2
ITER = int(T/Delta)
a = .1
b = 0.

def h(t,a=a): # impulse reponse
	if t < 0.:
		return 0.
	else:
		return np.exp(-a*t) 


def u(t): # input function u(t) = sin(t) * u_0(t)
	if t < 0.:
		return 0.
	else:
		return np.sin(t)


u_s = np.zeros(ITER)
h_s = np.zeros(ITER)

for n in range(ITER):
	u_s[n] = u(n * Delta)  
	h_s[n] = h(n * Delta)  


y_s = np.zeros(ITER)
	
for n in range(ITER):
	c = 0.  # Initialize the summation variable
	for m in range(n + 1):
		c += Delta * h_s[n - m] * u_s[m]  
	y_s[n] = c


f1,axs = plt.subplots(1,1)
axs.plot(np.arange(0,ITER)*Delta, y_s)
axs.set_xlabel("Time(s)")
axs.set_ylabel("Solution of ODE")
plt.show()
plt.close()