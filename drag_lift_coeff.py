import numpy as np
from numpy import pi 
import matplotlib.pyplot as plt


alpha1 = np.linspace(0, pi/6, 100)

CL1 = (9 / pi) * alpha1

plt.plot(alpha1, CL1)

alpha2 = np.linspace(pi/6, pi/2, 100)

CL2 = -(4.5/pi) * alpha2 + (4.5/2)

plt.plot(alpha2, CL2)

alpha3 = np.append(alpha1, alpha2)
CL3 = np.append(CL1, CL2)

# drag coefficient
CD = 0.05 + CL3**2 / (pi * 5);

plt.plot(alpha3, CD)

plt.show()

