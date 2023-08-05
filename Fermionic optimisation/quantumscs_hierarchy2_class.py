import scs
import numpy as np
from scipy import sparse
import math

# Setting up initial data and useful functions

# NB for a monomial with word of length 1 e.g. chi_1, input must be (1,)
class Monomial():
    def __init__(self, word: tuple):
        self.word = word

# Converts <chi^S psi, chi^T psi> to +-<psi, chi^symdif(S,T) psi> with the correct sign
# NB monomial words must be sorted
def q_mult(mon1: Monomial, mon2: Monomial):
    mon1word, mon2word = mon1.word, mon2.word
    resword = tuple(sorted(set(mon1word) ^ set(mon2word)))

    mon1word, mon2word = list(mon1word), list(mon2word)
    l = len(mon1word)
    count = int(l*(l-1)/2)
    while mon1word and mon2word:
        if mon1word[-1] <= mon2word[-1]:
            mon2word.pop()
        else:
            mon1word.pop()
            count += len(mon2word)
    
    return Monomial(resword), count%2

# Preliminary data for problem
class initialData():
    def __init__(self, n):
        self.n = n

        self.dim = int(1 + n*(n-1)/2)
        self.tot = int(n*(n-1)/2 + n*(n-1)*(n-2)*(n-3)/24)

class Basis():
    def __init__(self, data: initialData):
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

'''# Vec as above for sparse dok matrices (significantly slower than above function)
def sparseToVec(S):
    n = S.shape[0]
    S = sparse.dok_matrix.copy(S)
    S *= math.sqrt(2)
    S[range(n), range(n)] /= math.sqrt(2)
    return S[np.triu_indices(n)]'''

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
            if q_mult(monomials[i],monomials[j])[1] == q_mult(monomials[j],monomials[i])[1]:
                temp, sgn = q_mult(monomials[i], monomials[j])
                temp = temp.word

                index = order[temp]
                print(monomials[i].word, monomials[j].word, temp, sgn)
                if sgn == 0:
                    matrices[index][i,j] = 1
                    matrices[index][j,i] = 1
                else:
                    matrices[index][i,j] = -1
                    matrices[index][j,i] = -1

    # Converting these matrices to vector form (see required scs input format)
    vectors = [sparse.dok_matrix((math.comb(dim+1,2),1)) for _ in range(tot)]
    for i in range(tot):
        vectors[i] = sparse.csc_matrix(vec(matrices[i+1].toarray()).reshape(-1,1))
    A = sparse.csc_matrix(-sparse.hstack(vectors))
    print('Finished generating A')
    return A

def generate_b(data):
    dim = data.dim
    b = vec(np.identity(dim))
    print('Finished generating b')
    return b

def generate_c(data, basis):
    tot = data.tot
    order = basis.order
    monomials = basis.monomials

    # Constructing cost vector c
    c = np.zeros(tot)
    a_coeffs = np.random.randn(tot)
    for tuple, index in order.items():
        if len(tuple) == 4:
            c[index - 1] = a_coeffs[index - 1]
    c = np.hstack(c)
    print('Finished generating c')
    return c


# SDP problem class

class Problem():
    def __init__(self, data, basis):
        self.P = None
        self.A = generate_A(data, basis)
        self.b = generate_b(data)
        self.c = generate_c(data, basis)
        self.vars = dict(P=self.P, A=self.A, b=self.b, c=self.c)
        self.cone = dict(s=data.dim)

    def solve(self, eps_abs=1e-5, eps_rel=1e-5):
        solver = scs.SCS(self.vars, self.cone, eps_abs=eps_abs, eps_rel=eps_rel)
        return solver.solve()