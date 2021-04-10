import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d


def f(x,y):
    return (x+y**2)**2

fig = plt.figure()
ax = plt.axes(projection='3d')

x = np.linspace(-.5,.5,500)
y = np.linspace(-.5,.5,500)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
print("a")
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
#ax.plot3D(x,y,z)
plt.show()