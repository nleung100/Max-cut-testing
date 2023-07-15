import cvxpy as cp
import numpy as np

n = 10

L = np.random.randn(n,n)

Y = cp.Variable((n,n), symmetric=True)

objective = cp.Maximize(cp.trace(L @ Y))
constraints = []
constraints.append(Y >> 0)
for i in range(n):
    constraints.append(Y[i][i] == 1)

problem  = cp.Problem(objective, constraints)

problem.solve()

print('The optimal value is', problem.value)
print('An optimal Y is')
print(Y.value)

