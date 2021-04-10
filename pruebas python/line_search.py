import numpy as np

def f(x):
    suma = 0
    for i in range(len(x)):
        suma = suma + (i+1) * x[i]**2
    return suma    

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
    if resp:
        eigenValores = np.linalg.eig(hessiana(f, x, h))
        for i in range(n):
            if eigenValores[0][i] < h:
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

def encuentraMinimo(f,x, phi = .7, h =.0001):
    while not condicionesnecesarias(f, x, h):
        a = 1
        p = steepest(f, x)
        #p = quasi(f, x, h)

        while not condicionesWolfe(f, x, p, a):
            a = phi*a
        x = x + a*p
        print(x)
    return x
    


x = (1,2)
encuentraMinimo(f,x, .7,.0000001)

#print(condicionesWolfe(f, x, np.array((-1,-2)), 1))

#print(quasi(f,x))

#y = -x
#print(np.dot(x,y))
#print(condicionesnecesarias(f, (0, 0)))

