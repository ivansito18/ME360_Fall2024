import numpy as np
np.random.seed(123) # since we are working with random numbers, this command ensures the code gives the same output on every run


N = 100
L = 5.
Delta = (2*L)/(1.*N)

x = np.zeros(N)
y = np.zeros(N)

for i in range(0,N):
	x[i] = # insert code here
	y[i] = x[i] + np.random.normal(0.,1.,1)


## Least squares solution as obtained in HW4

yhat = # insert code here
xhat = # insert code here

w_opt = # insert code here
w0_opt = # insert code here