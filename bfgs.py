import numpy as np


def gradiente(f, x, h = .0001):
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        z = np.zeros(n)
        z[i] = h/2
        grad[i] = (f(x+z) - f(x-z))/h
    return grad

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

def condicionesnecesarias(f, x, h = .0001):
    resp = True
    n = len(x)
    grad = gradiente(f, x, h)
    for i in range(n):
        if abs(grad[i]) > h:
            resp = False
            break

    return resp        
####### direcciones de descenso ########
def steepest(f, x):
    return -gradiente(f,x)
def quasi(f, x, h=.0001):
    return -np.matmul(np.linalg.inv(hessiana(f,x, h)),gradiente(f,x,h))
    #return -np.linalg.solve(hessiana(f, x, h), gradiente(f, x, h))

def condicionesWolfe(f, x, p, a, c1=.5, c2=.9):
    resp = True
    if f(x + a*p) > f(x) + c1*a*np.dot(gradiente(f, x), p):
        resp=False
    if np.dot(gradiente(f, x + a*p), p) < c2*np.dot(gradiente(f, x), p):
        resp=False
    return resp

def generaAlpha(f, x, pk, c1=1e-4, c2 = 0.5, tol=1e-5):
    a = 1
    rho = 1/3
    gp = np.dot(gradiente(f,x), pk)
    while f(x + a*pk) > f(x) + c1*a*gp:
        a = a * rho
    return a

def BFGS_Hk(yk, sk, Hk):
    """
    Función que calcula La actualización BFGS de la matriz Hk
    In:
      yk: Vector n
      sk: Vector n
      Hk: Matriz nxn
    Out:
      Hk+1: Matriz nxn
    """
    n = len(yk)
    yk = np.array([yk]).T
    sk = np.array([sk]).T
    rhok = 1 / yk.T.dot(sk)
    Vk = (np.eye(n) - rhok * yk.dot(sk.T))
    Hk1 = Vk.T * Hk * Vk + rhok * sk.dot(sk.T)
    return Hk1

def BFGS(f, x0, tol = .0001, maxIter = 1000):
    xk = x0
    hk = hessiana(f, x0, tol)
    gk = gradiente(f,x0, tol)
    k = 0
    sk = 10
    while np.linalg.norm(gk) > tol and np.linalg.norm(sk) > tol and k < maxIter:
        pk = -np.dot(hk, gk)
        alpha = generaAlpha(f, xk, pk)
        xk1 = xk + alpha * pk
        sk = xk1 - xk
        Gk1 = gradiente(f, xk1)
        yk = Gk1 - gk
        hk = BFGS_Hk(yk, sk, hk)
        k += 1
        xk = xk1
        gk = Gk1
    return xk1, k


def encuentraMinimo(f,x, phi = .7, h =.0001):
    while not condicionesnecesarias(f, x, h):
        a = 1
        p = steepest(f, x)
        #p = quasi(f, x, h)

        while not condicionesWolfe(f, x, p, a):
            a = phi*a
        x = x + a*p
    return x

def f(x):
    suma = 0
    for i in range(len(x)):
        suma = suma + x[i]**2
    return suma

x0 = [1, 1, 1]
print(encuentraMinimo(f, x0))
print(BFGS(f, x0))