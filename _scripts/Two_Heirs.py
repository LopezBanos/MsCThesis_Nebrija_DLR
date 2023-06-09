#=============================================================================#
# Two-Heirs problem
# Author: Sergio Lopez Banos
# GitHub: /LopezBanos
# Citation: If you use this code in your project please cite the following 
# thesis:
'''
@Thesis{SerAlOr2023,
    author      = {Sergio López Baños},
    title       = {Transmission Expansion Planning by Quantum Annealing},
    type        = {diplomathesis}, 
    institution = {Nebrija University},
    year        = {2023},
}
'''
#=============================================================================#
#                            Importing packages                               #
#=============================================================================#
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

#=============================================================================#
#                                Ising Matrix                                  #
#=============================================================================#
# Two-Heirs problem parameters
v0=1 ; v1=3 ; v2=1
# Pauli Matrix and Identity
I = np.eye(2)
X = np.matrix([[0,1],[1,0]])
Z = np.matrix([[1,0],[0,-1]])

# Mapping of our problem into Ising
Ising = np.matrix([[0, 2*v0*v1, 2*v0*v2],[0, 0, 2*v1*v2],[0,0,0]])

#=============================================================================#
#                                OUTER PRODUCTS                               #
#=============================================================================#
# Z
Z0Z1 = np.kron(Z, np.kron(Z, I))
Z0Z2 = np.kron(Z, np.kron(I, Z))
Z1Z2 = np.kron(I, np.kron(Z, Z))

# X
X0 = np.kron(X, np.kron(I, I))
X1 = np.kron(I, np.kron(X, I))
X2 = np.kron(I, np.kron(I, X))


#=============================================================================#
#                                 Hamiltonians                                #
#=============================================================================#
H_Ising = Ising[0,1]*Z0Z1 + Ising[0,2]*Z0Z2 + Ising[1,2]*Z1Z2
H_ini = -X0 - X1 - X2

# Check eigenvectors of Ising Hamiltonian
w, v = LA.eig(H_Ising)
print("eigenvalues:", w)
min_indices = np.where(w == w.min())[0] 
print("min_eigenvalues indices:", min_indices[0])
print("The associated eigenvectors")
for i in range(len(min_indices)):
    print(v[min_indices[i]])

T = 1
time = np.linspace(0, T, 100)
#H = (1-t/T)*H_ini + (t/T)*H_Ising
eigen_H = []
eigen_H_Ising = []
eigen_H_ini = []
for t in time:
    # Compute the Hamiltonian at that snapshot
    H = (1-t/T)*H_ini + (t/T)*H_Ising

    # Compute the ground eigenstate
    # Total Hamiltonian
    w, v = LA.eig(H)
    min_index = np.argmin(w)
    eigen_H.append(w[min_index])
    # Ising Hamiltonian
    w, v = LA.eig((t/T)*H_Ising)
    min_index = np.argmin(w)
    eigen_H_Ising.append(w[min_index])
    # Initial Hamiltonian
    w, v = LA.eig((1-t/T)*H_ini)
    min_index = np.argmin(w)
    eigen_H_ini.append(w[min_index])

#=============================================================================#
# Plot with your preferred framework                                   
#=============================================================================#

