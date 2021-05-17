import numpy as np
from numpy.compat import py3k

def dirConj(x0, A, b):
    n = len(x0)
    print(n)
    print(x0)
    ps = []
    r = []
    alpha = np.zeros(n)
    print(alpha)
    for i in range(n):
        z = np.zeros(n)
        z[i] = 1
        ps = ps + [z]
    print(ps)
    for i in range(n):
        r = r + [np.dot(A, x0) - b]
        print(r[i].T)
        print(type(-np.dot(r[i].T, ps[i])))
        alpha[i] = -np.dot(r[i].T, ps[i])/np.dot(np.dot(ps[i],A),ps[i])
        print(x0)
        print(alpha[i]*ps[i])
        x0= x0 + (alpha[i]*ps[i]).T
        
    return x0


def gradConj(x0, A, b):
    rk = np.dot(A,x0) - b
    pk = -rk
    xk = x0
    print(rk)
    while np.dot(rk, rk.T) > .001:
        alphak = -np.dot(rk, rk.T)/np.dot(pk, A.dot(pk.T))
        print(alphak[0])
        xk_1 = xk + alphak*pk
        rk_1 = rk + alphak[0]*A.dot(pk.T).T
        print(rk_1)
        bk = np.dot(rk_1, rk_1.T)/np.dot(rk, rk.T)
        pk_1 = -rk_1+ bk*pk
        rk = rk_1
        print(rk)
        pk = pk_1
        print(pk)
        xk = xk_1
        print(xk)
    return xk








A = np.matrix('4 0 0; 0 2 0; 0 0 3')
print(A)
b = np.matrix('4; 2; 3')
print(b.shape)
x0 = np.matrix('1; 1; 1')

x1 = dirConj(x0,A,b)
#print(np.dot(A,x1))
x = gradConj(x0,A,b)
#print(np.dot(A,x))




