import numpy as np

def gradiente(f, x, h = .0001):
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        z = np.zeros(n)
        z[i] = h/2
        grad[i] = (f(x+z) - f(x-z))/h
    return grad

def f(x):
    n = len(x)
    y = 0
    for i in range(n):
        y = y + x[i]**2
    return y

def hessiana(f, x, h = .0001):
    n = len(x)
    hess = np.zeros((n,n))
    for i in range(n):
        w = np.zeros(n)
        w[i] = h
        for j in range(n):
            if i==j:
                hess[i][j] = (-f(x+2*w) +16*f(x+w) - 30*f(x) + 16*f(x-w) -f(x-2*w))/(12*h**2)
            else:
                z = np.zeros(n)
                z[j] = h
                hess[i][j] = (f(x + w + z) - f(x - w + z) - f(x - z + w) + f(x - z - w))/(4*h**2)
    return hess

def g(x):
    n = len(x)
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            A[i][j] = i + j
    return np.matmul(np.matmul(np.transpose(x), A), x)


#x = [1, 1, 1, 1]
#print(g(x))

#y = gradiente(f, x)
#y = hessiana(g, x)
#print(y)


#2.3
#hessiana(g, x)



