import cvxpy as cp
import numpy as np
import math

n = 4

L = np.random.randn(n,n)

dim = 1 + math.comb(n,2)

Y = cp.Variable((dim,dim), symmetric=True)

constraints = []
constraints.append(Y >> 0)
for i in range(dim):
    constraints.append(Y[i][i] == 1)

monomials = [[]]
for i in range(1,n+1):
    for j in range(i+1,n+1):
        monomials.append([i,j])

fours = {}
for i in range(dim):
    for j in range(i+1,dim):
        temp = np.sort(np.concatenate((np.setdiff1d(monomials[i],monomials[j]), np.setdiff1d(monomials[j], monomials[i]))))
        if len(temp) == 2:
            a = temp[0]
            b = temp[1]
            constraints.append(Y[i][j] == Y[0][int((a - 1)*n - a*(a-1)/2 + b - a)])
        if len(temp) == 4:
            if tuple(temp) in fours.keys():
                constraints.append(Y[i][j] == Y[fours[tuple(temp)][0]][fours[tuple(temp)][1]])
            else:
                fours.update({tuple(temp): (i,j)})


sum = 0
for i in range(n):
    for j in range(n):
        sum += Y[0][int((i - 1)*n - i*(i-1)/2 + j - i)] * L[i][j]

objective = cp.Maximize(sum)

problem  = cp.Problem(objective, constraints)

problem.solve()

print('The optimal value is', problem.value)
print('An optimal Y is')
print(Y.value)