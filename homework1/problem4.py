import numpy as np
import matplotlib.pyplot as plt

FONTSIZE = 16

def signal(time):
    y = np.zeros(time.shape)
    for i, t in enumerate(time):
        if t >= 0:
            y[i] = (np.sin(t))**3
    return y

t = np.linspace(-3, 22, 1000)
y = signal(t)

plt.figure(figsize=(10, 6))
plt.title(
    'Plot of $\\text{sin}^3(t)$ for $t > 0$, 0 otherwise', 
    fontsize=FONTSIZE+2
)
plt.xlabel('$t$', fontsize=FONTSIZE)
plt.ylabel('$y$', fontsize=FONTSIZE)
plt.grid(True)
plt.axhline(y=0, color='black', linestyle='-', alpha=1)
plt.axvline(x=0, color='black', linestyle='-', alpha=1)
plt.plot(t, y, linewidth=5)
plt.xlim(-2, 21)
plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
plt.show()