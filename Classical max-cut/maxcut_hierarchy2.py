import cvxpy as cp
import numpy as np
import math
import time

start = time.time()

# Setting up initial data of problem
n = 15

L = np.random.randn(n,n)
L = L + L.T

dim = 1 + math.comb(n,2)

Y = cp.Variable((dim,dim), symmetric=True)

# Constructing diaognal constraints on matrix (y_emptyset = 1)
constraints = []
constraints.append(Y >> 0)
for i in range(dim):
    constraints.append(Y[i][i] == 1)

# Using degree 2 monomials to index matrix
monomials = [[]]
for i in range(1,n+1):
    for j in range(i+1,n+1):
        monomials.append([i,j])

# Constructing symmetric difference constraints on matrix entries (y_symdiff(U,V))
fours = {}
for i in range(dim):
    for j in range(i+1,dim):
        temp = np.sort(np.concatenate((np.setdiff1d(monomials[i],monomials[j]), np.setdiff1d(monomials[j], monomials[i]))))
        if len(temp) == 2:
            a = temp[0]
            b = temp[1]
            constraints.append(Y[i][j] == Y[0][int((a - 1)*n - a*(a-1)/2 + b - a)])
        if len(temp) == 4:
            if tuple(temp) in fours:
                constraints.append(Y[i][j] == Y[fours[tuple(temp)][0]][fours[tuple(temp)][1]])
            else:
                fours.update({tuple(temp): (i,j)})

# Constructing objective function
sum = 0
for i in range(1,n+1):
    for j in range(1,n+1):
        ri = min(i,j)
        rj = max(i,j)

        if i == j:
            sum += L[i-1][j-1]
        else:
            sum += Y[0][int((ri - 1)*n - ri*(ri-1)/2 + rj - ri)] * L[i-1][j-1]

print('Initialisation took', time.time() - start, 's')

# Solving and processing
objective = cp.Maximize(sum)

problem1  = cp.Problem(objective, constraints)

problem1.solve(solver=cp.SCS, verbose=True)


print('The optimal value is', problem1.value)
print('An optimal Y is')
print(Y.value)
print('Solve time for SCS is', time.time() - start, 's')