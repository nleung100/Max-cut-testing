import scs
import numpy as np
import scipy as sp
from scipy import sparse
import math
import time

start = time.time()

# The vec function as documented in api/cones (weird scs input format)
def vec(S):
    n = S.shape[0]
    S = np.copy(S)
    S *= np.sqrt(2)
    S[range(n), range(n)] /= np.sqrt(2)
    return S[np.triu_indices(n)]

# The mat function as documented in api/cones (weird scs input format)
def mat(s):
    n = int((np.sqrt(8 * len(s) + 1) - 1) / 2)
    S = np.zeros((n, n))
    S[np.triu_indices(n)] = s / np.sqrt(2)
    S = S + S.T
    S[range(n), range(n)] /= np.sqrt(2)
    return S

# Setting up initial data of problem
n = 15
nc2 = math.comb(n,2)
nc4 = math.comb(n,4)

L = np.random.randn(n,n)
L = L + L.T

dim = 1 + math.comb(n,2)
tot = math.comb(n,2) + math.comb(n,4)

# Using degree 2 monomials to index matrix
monomials = [[]]
for i in range(1,n+1):
    for j in range(i+1,n+1):
        monomials.append([i,j])

# Assigning an order to these monomials
order = {}
for i in range(1,n+1):
    for j in range(i+1,n+1):
        order[(i,j)] = len(order)+1
for i in range(1,n+1):
    for j in range(i+1,n+1):
        for k in range(j+1,n+1):
            for l in range(k+1,n+1):
                order[(i,j,k,l)] = len(order)+1

# Vector b in Ax + b = s constraint/A_0 matrix in LMI form (see required scs input format)
B = np.identity(dim)
b = vec(B)

# Constructing A_1, ..., A_k matrices in LMI form
matrices = [np.zeros((dim,dim)) for _ in range(tot+1)]
matrices[0] = B
for i in range(dim):
    for j in range(i+1,dim):
        temp = np.sort(np.concatenate((np.setdiff1d(monomials[i],monomials[j]), np.setdiff1d(monomials[j], monomials[i]))))
        if len(temp) == 2:
            p = temp[0]
            q = temp[1]
            index = int((p - 1)*n - p*(p-1)/2 + q - p)
            matrices[index][i][j] = 1
            matrices[index][j][i] = 1
        if len(temp) == 4:
            p = temp[0]
            q = temp[1]
            r = temp[2]
            s = temp[3]
            index = order[(p,q,r,s)]
            matrices[index][i][j] = 1
            matrices[index][j][i] = 1

# Converting these matrices to vector form (see required scs input format)
vectors = [np.zeros(math.comb(dim+1,2)) for _ in range(tot)]
for i in range(tot):
    vectors[i] = vec(matrices[i+1]).reshape(-1,1)
result = np.hstack(vectors)

sparse_result = sparse.csc_matrix(result)
A = sparse.csc_matrix(-np.hstack(vectors))

# Constructing cost vector c (actually -c since scs minimises and we want to maximise - see required scs input format)
c = np.zeros(tot)
for tuple, index in order.items():
    if len(tuple) == 2:
        i = tuple[0]
        j = tuple[1]
        c[index - 1] = 2*L[i-1][j-1]
c = np.hstack([-c])

# No quadratic part in objective function for this problem
P = None

# Solving and processing
data = dict(P=P, A=A, b=b, c=c)
cone = dict(s=dim)

solver = scs.SCS(data, cone, eps_abs=1e-5, eps_rel=1e-5)
print('Solving')
sol = solver.solve()

diags = 0
for i in range(n):
    diags += L[i][i]
finalobj = -sol['info']['pobj'] + diags

print('The optimal value is', finalobj)
print('Solve time for SCS is', time.time() - start, 's')