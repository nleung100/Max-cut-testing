import cvxpy as cp
import numpy as np
import time

start = time.time()

# Setting up initial data of problem
n = 100

L = np.random.randn(n,n)
L = L + L.T


Y = cp.Variable((n,n), symmetric=True)

# Constructing diagonal constraints on matrix (y_emptyset = 1)
constraints = []
constraints.append(Y >> 0)
for i in range(n):
    constraints.append(Y[i][i] == 1)

# Solving and processing
objective = cp.Maximize(cp.trace(L @ Y))

problem  = cp.Problem(objective, constraints)

problem.solve()

print('The optimal value is', problem.value)
print('An optimal Y is')
print(Y.value)
print('Solve time for Mosek is', time.time() - start, 's')