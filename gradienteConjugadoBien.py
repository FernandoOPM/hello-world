import numpy as np

def dirConj(x0, A, b):
    n = len(x0)
    pk = []
    for i in range(n):
        z = np.zeros(n)
        z[i] = 1
        pk = pk + [z]
    for i in range(n):
        r = np.dot(A, x0) -b
        alpha = -np.dot(r, pk[i])/np.dot(np.dot(pk[i], A), pk[i])
        x0= x0 + alpha*pk[i]
    return x0

def gradConj(x0, A, b):
    rk = np.dot(A, x0) -b
    pk = -rk
    xk = x0
    cont =0
    while np.dot(rk,rk) >.0001 and cont<1000:
        ak = np.dot(rk, rk)/np.dot(np.dot(pk, A), pk)
        xk_1 = xk + ak*pk
        rk_1 = rk + ak*np.dot(A, pk)
        bk_1 = np.dot(rk_1, rk_1)/np.dot(rk, rk)
        pk_1 = -rk_1 + bk_1*pk
        xk = xk_1
        pk = pk_1
        rk = rk_1
        cont = cont + 1
    return xk

def gradConjPrecond(x0, A, b, M):
    rk = np.dot(A, x0) -b
    yk = np.linalg.solve(M, rk)
    pk = -yk
    xk = x0
    while np.dot(rk,rk) >.0001:
        print("rk: ")
        print(rk)
        ak = np.dot(rk, yk)/np.dot(np.dot(pk, A), pk)
        xk_1 = xk + ak*pk
        rk_1 = rk + ak*np.dot(A, pk)
        yk_1 = np.linalg.solve(M, rk_1)
        print("yk+1:")
        print(yk_1)
        bk_1 = np.dot(rk_1, yk_1)/np.dot(rk, yk)
        pk_1 = -yk_1 + bk_1*pk
        print(pk_1)
        yk = yk_1
        xk = xk_1
        pk = pk_1
        rk = rk_1
    return xk


A = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
x0 = [0,0,0]
b = [1,1,1]

xresp = np.linalg.solve(A,b)
w, M = np.linalg.eig(A)
print(w)
print(M)
print(np.matmul(M,np.linalg.inv(M)))
print("xresp: ")
print(xresp)
"""
x = dirConj(x0,A,b)
print("direcciones conjudadas: ")
print(x)
x1 = gradConj(x0, A, b)
print("gradiente conjugado: ")
print(x1)
x2 = gradConjPrecond(x0, A, b, M)
print("gradiente conjugado precondicionado: ")
print(x2)
"""