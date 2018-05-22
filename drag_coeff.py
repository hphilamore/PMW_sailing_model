import numpy as np
import matplotlib.pyplot as plt
a = [0, 10, 20, 30]
Cd = [-0.023, 0.044, 0.179, 0.354]
Cl = [-0.047, 0.374, 0.623, 0.756]

c, d, e = np.polyfit(a, Cd, 2)
Cdfit = np.poly1d([c, d, e])(a)
CL=np.array(Cl)


plt.plot(a, Cd, label= f'raw')
plt.plot(a, Cdfit, label= f'2nd order ')

# add a legend
plt.legend(loc='best')

print(c, d, e)

plt.show()



c, d, e = np.polyfit(Cl, Cd, 2)
Cdfit = np.poly1d([c, d, e])(Cd)

CL=np.array(Cl)
CD = 0.05 + CL**2 / (np.pi * 5);


plt.plot(Cl, Cd, label= f'raw')
plt.plot(Cl, Cdfit, label= f'2nd order ')
plt.plot(CL, CD, label= f'theoretical')

# add a legend
plt.legend(loc='best')

print(c, d, e)

plt.show()

c, d, e = np.polyfit(a, Cl, 2)
Clfit = np.poly1d([c, d, e])(a)
A=np.array(a)
CD = 0.05 + CL**2 / (np.pi * 1);
plt.plot(A, CD, label= f'theoretical')



plt.plot(a, Cl, label= f'raw')
plt.plot(a, Clfit, label= f'2nd order ')

# add a legend
plt.legend(loc='best')

print(c, d, e)

plt.show()