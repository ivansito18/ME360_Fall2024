import numpy as np
np.random.seed(123) # since we are working with random numbers, this command ensures the code gives the same output on every run

import matplotlib.pyplot as plt

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


## function to find the value and gradient of J

def J(x,y,w,w0,N):
	return (1./2.*N)*(np.linalg.norm(y - w*x - w0*np.ones(N))**2)


def gradJw(x,y,w,w0,N): # gradient of J with respect to w
	return (-1./N)*np.sum(np.multiply(y - w*x - w0*np.ones(N),x))

def gradJw0(x,y,w,w0,N): # gradient of J with respect to w0
	return (-1./N)*np.sum(y - w*x - w0*np.ones(N))

## implementation of gradient descent

EPOCHS = 50 # we will do 1000 steps of gradient descent
alpha = 0.01*np.ones(EPOCHS+1) # coefficient of gradient in gradient descent step

# arrays to store history of w and w0 as gradient descent occurs
w_hist = np.zeros(EPOCHS+1)
w0_hist = np.zeros(EPOCHS+1)
J_hist = np.zeros(EPOCHS+1)
J_hist[0] = J(x,y,w_hist[0],w0_hist[0],N)

for epoch in range(0,EPOCHS):
	wepoch = w_hist[epoch]
	w0epoch = w0_hist[epoch]

	wnew = # insert code here
	w0new = # insert code here

	w_hist[epoch+1] = wnew
	w0_hist[epoch+1] = w0new
	J_hist[epoch+1] = J(x,y,wnew,w0new,N)


J_opt = J(x,y,w_opt,w0_opt,N)

f1, axs = plt.subplots(1,1)
axs.plot(np.arange(EPOCHS+1),J_hist,c="blue",ls="solid")
axs.axhline(J_opt,c="black",ls="dashed")
axs.set_yscale('log')
plt.show()


