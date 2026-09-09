import numpy as np
import matplotlib.pyplot as plt

x= np.linspace(0,1,20)

def mystery(x, *c):
	return c[0] + c[1]*x + c[2]*x**2 + c[3]*x**3 + c[4]*x**4 + c[5]*x**5

c = [0.01, 2.12, -2.12, 0.5, -0.9 , 1.4]
y = mystery(x,*c)

#np.savetxt('MysteryPolynomial-XY.txt', (x, y))
np.savetxt('Data-XY.txt', (x, y))

plt.plot( x, y ,'ro',label='y=?')
plt.legend()
plt.title('Mystery Polynomial')
plt.savefig('mystery.png')
plt.clf()