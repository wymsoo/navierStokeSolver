import numpy as np
import matplotlib.pyplot as plt
import os

sd = 1
mean = 0

k = 1/np.sqrt(2*np.pi*sd)

def f0(y):
    return k*np.exp(-(y**2/2/sd**2))
def f1(y):
    return k*np.exp(-((y-1)**2/2/sd**2))

y = np.arange(-5,5,0.01)
f0y= f0(y)
f1y = f1(y)
x = np.ones(100) * 1/2

plt.plot(y,f0y,label='f(y|H=0)')
plt.plot(y,f1y, label='f(y|H=1)')
plt.xlabel('y')
plt.ylabel('f(y|H)')
plt.vlines(1/2, 0,0.5, label='y=1/2', colors='black')
plt.legend()
plt.title('Plots of two distributions of the observation y given H')
plt.show()

