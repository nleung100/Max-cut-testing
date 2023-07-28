import openfermion as of
import itertools
import numpy as np
import scipy as sp

n = 4   # Is even in SYK model
q = 2   # Only physically natural if q even

# Create tuple basis for Majorana operators
majoranaTuples = list(itertools.combinations([i for i in range(1,n+1)], q))
print(majoranaTuples)
print('')

# Initialise coeffs a_S from IID standard Gaussian random variables
a_coeffs = np.random.randn(len(majoranaTuples))
print(a_coeffs)
print('')

# Create sum of Majorana operators to find max eigenvalue of
hamiltonian = of.ops.MajoranaOperator()
for i in range(len(majoranaTuples)):
    hamiltonian += of.ops.MajoranaOperator((majoranaTuples[i]), a_coeffs[i])

# Include prefactor of i^(q choose 2) to ensure hamiltonian is self-adjoint
hamiltonian *= (1.j)**(q*(q-1)/2)
print(hamiltonian)
print('')

# Convert hamiltonian to sparse matrix form (have to convert to FermionOperator type first) and check self-adjointedness
mat_repr = of.linalg.get_sparse_operator(of.transforms.get_fermion_operator(hamiltonian))
print(mat_repr.toarray())
print(sp.linalg.ishermitian(mat_repr.toarray()))
print('')

# Solve for ground state energy and eigenstate
energy, state = of.linalg.get_ground_state(mat_repr)
print(energy)
print(state)