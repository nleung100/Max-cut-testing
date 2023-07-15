import cvxpy as cp
import numpy as np
import math

n = 4

L = np.random.randn(n,n)
L = L + L.T
print('Laplacian L_G is')
print(L)

Y1 = cp.Variable((n,n), symmetric=True)

objective1 = cp.Maximize(cp.trace(L @ Y1))
constraints1 = []
constraints1.append(Y1 >> 0)
for i in range(n):
    constraints1.append(Y1[i][i] == 1)

problem1  = cp.Problem(objective1, constraints1)

problem1.solve()

print('The optimal value for hierarchy 1 is', problem1.value)
print('An optimal Y is')
print(Y1.value)

dim = 1 + math.comb(n,2)

Y2 = cp.Variable((dim,dim), symmetric=True)

constraints2 = []
constraints2.append(Y2 >> 0)
for i in range(dim):
    constraints2.append(Y2[i][i] == 1)

monomials = [[]]
for i in range(1,n+1):
    for j in range(i+1,n+1):
        monomials.append([i,j])

fours = {}
for i in range(1,dim):
    for j in range(i+1,dim):
        temp = np.sort(np.concatenate((np.setdiff1d(monomials[i],monomials[j]), np.setdiff1d(monomials[j], monomials[i]))))
        if len(temp) == 2:
            a = temp[0]
            b = temp[1]
            constraints2.append(Y2[i][j] == Y2[0][int((a - 1)*n - a*(a-1)/2 + b - a)])
        if len(temp) == 4:
            if tuple(temp) in fours.keys():
                constraints2.append(Y2[i][j] == Y2[fours[tuple(temp)][0]][fours[tuple(temp)][1]])
            else:
                fours.update({tuple(temp): (i,j)})


sum = 0
for i in range(1,n+1):
    for j in range(1,n+1):
        ri = min(i,j)
        rj = max(i,j)

        if i == j:
            sum += L[i-1][j-1]
        else:
            sum += Y2[0][int((ri - 1)*n - ri*(ri-1)/2 + rj - ri)] * L[i-1][j-1]

objective2 = cp.Maximize(sum)

problem2  = cp.Problem(objective2, constraints2)

problem2.solve()

print('The optimal value for hierarchy 2 is', problem2.value)
print('An optimal Y is')
print(Y2.value)

print(problem1.value - problem2.value)