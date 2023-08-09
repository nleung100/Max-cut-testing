import scs
import numpy as np
from scipy import sparse
import math

# Setting up initial data and useful functions

# NB for a monomial with word of length 1 e.g. chi_1, input must be (1,)
class Monomial():
    def __init__(self, word: tuple):
        self.word = word

def symdif(mon1: Monomial, mon2: Monomial):
    mon1word, mon2word = mon1.word, mon2.word
    resword = tuple(sorted(set(mon1word) ^ set(mon2word)))
    return Monomial(resword)

# Preliminary data for problem
class InitialData():
    def __init__(self, n, L=None):
        self.n = n
        if L is not None:
            self.L = L
        else:
            tempL = np.random.randn(n,n)
            tempL = tempL + tempL.T
            self.L = tempL
        self.dim = int(1 + n*(n-1)/2)
        self.tot = int(n*(n-1)/2 + n*(n-1)*(n-2)*(n-3)/24)

class MonomialBasis():
    def __init__(self, data: InitialData):
        n = data.n

        monomials = [Monomial(())]
        for i in range(1,n+1):
            for j in range(i+1,n+1):
                monomials.append(Monomial((i,j)))
        self.monomials = monomials

        order = {}
        for i in range(1,n+1):
            for j in range(i+1,n+1):
                order[(i,j)] = len(order)+1
        for i in range(1,n+1):
            for j in range(i+1,n+1):
                for k in range(j+1,n+1):
                    for l in range(k+1,n+1):
                        order[(i,j,k,l)] = len(order)+1
        self.order = order

# The vec function as documented in api/cones (weird scs input format)
def vec(S):
    n = S.shape[0]
    S = np.copy(S)
    S *= np.sqrt(2)
    S[range(n), range(n)] /= np.sqrt(2)
    return S[np.triu_indices(n)]

# Vec as above for sparse matrices (only works for matrices with no diagonal elements) - much faster than 
def sparseToVec(S):
    n = S.shape[0]

    # Obtaining lower triangular elements
    S = sparse.tril(sparse.coo_matrix(S, copy=True), -1)

    # Creating coo format sparse vector of flattened matrix
    veclen = int(n*(n+1)/2)
    rows = [int(n*(n+1)/2 - (n-c)*(n-c+1)/2 + r-c) for r,c in zip(S.row, S.col)]
    cols = [0]*S.nnz
    data = []
    for k in range(len(S.data)):
        if S.row[k] == S.col[k]:
            data.append(S.data[k])
        else:
            data.append(math.sqrt(2)*S.data[k])
    M = sparse.coo_matrix((data, (rows, cols)), shape = (veclen, 1))
    return M

# The mat function as documented in api/cones (weird scs input format)
def mat(s):
    n = int((np.sqrt(8 * len(s) + 1) - 1) / 2)
    S = np.zeros((n, n))
    S[np.triu_indices(n)] = s / np.sqrt(2)
    S = S + S.T
    S[range(n), range(n)] /= np.sqrt(2)
    return S


# Functions for generating SDP for level 2 hierarchy
def generate_A(data, basis):
    dim = data.dim
    tot = data.tot
    monomials = basis.monomials
    order = basis.order

    matrices = [sparse.dok_matrix((dim,dim)) for _ in range(tot+1)]
    for i in range(dim):
        for j in range(i+1,dim):
            temp = symdif(monomials[i], monomials[j]).word
            index = order[temp]
            matrices[index][i,j] = 1
            matrices[index][j,i] = 1

    # Converting these matrices to vector form (see required scs input format)
    vectors = [sparse.csc_matrix((math.comb(dim+1,2),1)) for _ in range(tot)]
    for i in range(tot):
        vectors[i] = sparse.csc_matrix(sparseToVec(matrices[i+1]))
    A = sparse.csc_matrix(-sparse.hstack(vectors))
    print('Finished generating A')
    return A


def generate_b(data):
    dim = data.dim
    b = vec(np.identity(dim))
    print('Finished generating b')
    return b

def generate_c(data, basis):
    L = data.L
    tot = data.tot
    order = basis.order

    # Constructing cost vector c (actually -c since scs minimises and we want to maximise - see required scs input format)
    c = np.zeros(tot)
    for tuple, index in order.items():
        if len(tuple) == 2:
            i = tuple[0]
            j = tuple[1]
            c[index - 1] = 2*L[i-1][j-1]
    c = np.hstack([-c])
    print('Finished generating c')
    return c


# SDP problem class

class SDPRelaxation():
    def __init__(self, data: InitialData):
        self.data = data
        self.basis = MonomialBasis(data)

        self.P = None
        self.A = None
        self.b = None
        self.c = None
    
    def generate_A(self):
        self.A = generate_A(self.data, self.basis)
    
    def generate_b(self):
        self.b = generate_b(self.data)
    
    def generate_c(self):
        self.c = generate_c(self.data, self.basis)

    def build(self):
        self.A = generate_A(self.data, self.basis)
        self.b = generate_b(self.data)
        self.c = generate_c(self.data, self.basis)
        self.vars = dict(P=self.P, A=self.A, b=self.b, c=self.c)
        self.cone = dict(s=self.data.dim)

    def solve(self, eps_abs=1e-5, eps_rel=1e-5):
        solver = scs.SCS(self.vars, self.cone, eps_abs=eps_abs, eps_rel=eps_rel)
        return solver.solve()