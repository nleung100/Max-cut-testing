import numpy as np
import scipy as sp
from scipy import sparse
import math
import itertools
import scs
import openfermion as of

#Forced q=4, n>=4 and even
#Coefficient vector a is of length (n choose 4)

# Setting up initial data and useful functions

# NB for a monomial with word of length 1 e.g. chi_1, input must be (1,)
class Monomial():
    def __init__(self, word: list):
        self.word = word

# Converts <chi^S psi, chi^T psi> to +-<psi, chi^symdif(S,T) psi> with the correct sign
# NB monomial words must be sorted
def q_mult(mon1: Monomial, mon2: Monomial):
    # Creating resultant chi^symdif(S,T) monomial
    mon1word, mon2word = mon1.word, mon2.word
    resword = sorted(set(mon1word) ^ set(mon2word))

    # Calculating the correct sign - this is done by performing a merge sort on mon1word, mon2word whilst counting the number of inversions in the array
    l = len(mon1word)
    count = int(l*(l-1)/2)

    i,j = len(mon1word) - 1, len(mon2word) - 1
    curlen = len(mon2word)
    while i >= 0 and j >= 0:
        if mon1word[i] <= mon2word[j]:
            j -=1
            curlen -=1
        else:
            i -=1
            count += curlen
    
    return Monomial(resword), count%2

# Preliminary data for problem
class InitialData():
    def __init__(self, n, l, a):
        self.n = n
        self.l = l
        self.dim = 0
        for i in range(l+1):
            self.dim += int(math.comb(n,i))
        self.tot = int(math.comb(n,4))
        self.a = a

class MonomialBasis():
    def __init__(self, data: InitialData):
        n = data.n
        l = data.l

        arr = [i for i in range(1,n+1)]

        monomials = [Monomial([])]
        for i in range(1,l+1):
            mons = itertools.combinations(arr, i)
            for mon in mons:
                monomials.append(Monomial(list(mon)))
        self.monomials = monomials

        # Can replace order dictionary with a formula for monomial order
        order = {}
        mons = itertools.combinations(arr, 4)
        for mon in mons:
            order[mon] = len(order) + 1

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
            temp, sgn = q_mult(monomials[i],monomials[j])
            if len(temp.word)%4 == 0:
                temp = tuple(temp.word)

                index = order[temp]
                if sgn == 0:
                    matrices[index][i,j] = 1
                    matrices[index][j,i] = 1
                else:
                    matrices[index][i,j] = -1
                    matrices[index][j,i] = -1

    # Converting these matrices to vector form (see required scs input format)
    vectors = [sparse.csc_matrix((math.comb(dim+1,2),1)) for _ in range(tot)]
    for i in range(tot):
        vectors[i] = sparse.csc_matrix(sparseToVec(matrices[i+1]))
    A = sparse.csc_matrix(-sparse.hstack(vectors))
    '''print('Finished generating A')'''
    return A

def generate_b(data):
    dim = data.dim
    b = vec(np.identity(dim))
    '''print('Finished generating b')'''
    return b

def generate_c(data, basis):
    tot = data.tot
    order = basis.order
    a = data.a

    # Constructing cost vector c
    c = np.zeros(tot)
    for tuple, index in order.items():
        if len(tuple) == 4:
            c[index - 1] = a[index - 1]
    c = np.hstack(c)
    '''print('Finished generating c')'''
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
        solver = scs.SCS(self.vars, self.cone, eps_abs=eps_abs, eps_rel=eps_rel, verbose=False)
        return solver.solve()


def exact_diag(n, a):
    # Create tuple basis for Majorana operators
    majoranaTuples = list(itertools.combinations([i for i in range(1,n+1)], 4))
    '''print(majoranaTuples)
    print('')'''

    # Initialise coeffs a_S from IID standard Gaussian random variables
    #a_coeffs = np.random.randn(len(majoranaTuples))
    a_coeffs = a
    '''print(a_coeffs)
    print('')'''

    # Create sum of Majorana operators to find max eigenvalue of
    hamiltonian = of.ops.MajoranaOperator()
    for i in range(len(majoranaTuples)):
        hamiltonian += of.ops.MajoranaOperator((majoranaTuples[i]), a_coeffs[i])

    # Include prefactor of i^(q choose 2) to ensure hamiltonian is self-adjoint
    #hamiltonian *= (1.j)**(q*(q-1)/2)
    '''print(hamiltonian)
    print('')'''

    # Convert hamiltonian to sparse matrix form (have to convert to FermionOperator type first) and check self-adjointedness
    mat_repr = of.linalg.get_sparse_operator(of.transforms.get_fermion_operator(hamiltonian))
    '''print(np.shape(mat_repr.toarray()))
    print(sp.linalg.ishermitian(mat_repr.toarray()))
    print('')'''

    # Solve for ground state energy and eigenstate
    energy, state = of.linalg.get_ground_state(mat_repr)
    print(energy)
    '''print(state)'''

def relax(n,l,a):
    data = InitialData(n,l,a)
    prob = SDPRelaxation(data)
    prob.build()
    print(prob.solve()['info']['pobj'])