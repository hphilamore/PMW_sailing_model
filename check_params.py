import numpy as np
from numpy import exp
import matplotlib.pyplot as plt


tc = np.linspace(0,100, 100)
tc /= 100
asp = np.linspace(0,40, 100)



F1 = 1.190 * (1.0-(tc)**2)
F2 = 0.65 + 0.35 * exp(-(9.0/asp)**2.3)
G1 = 2.3 * exp(-(0.65 * (tc))**0.9)
G2 = 0.52 + 0.48 * exp(-(6.5/asp)**1.1)


fig = plt.subplots()
plt.plot(asp, F1*G1, label = 'F1*G1/asp')
plt.legend()
fig = plt.subplots()
plt.plot(asp, F2*G2, label = 'F2*G2/asp')
plt.legend()
fig = plt.subplots()
plt.plot(tc, G1*F1, label = 'G1*F1/tc')
plt.legend()
fig = plt.subplots()
plt.plot(tc, G2*F2, label = 'G2*F2/tc')
plt.legend()
plt.show()