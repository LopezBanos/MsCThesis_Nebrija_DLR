#=============================================================================#
# Transmission Expansion Planning (TEP) Problem: 
# A Three Node Fully Connected Network
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


#=============================================================================#
#                            Importing packages                               #
#=============================================================================#
import dimod
import neal
import numpy as np     
import pandas as pd    
from dimod import ConstrainedQuadraticModel,BinaryQuadraticModel, Binary
from dimod import Integer, quicksum


#=============================================================================#
#                            Fill up the coordinates                          #
#=============================================================================#      
C_iv = np.array([10, 20, 30], dtype=int)     # Investment Cost Coefficients
C_oc = np.array([10, 5, 2], dtype=int)       # Operational Cost Coefficients
g_max = np.array([10, 10, 10], dtype=int)    # Generator nominal power 
S =  int(np.shape(C_iv)[0])                  # Number of rows of the matrix
D = np.array([10, 12, 10], dtype=int)        # Demand at each node 


#=============================================================================#
#                               Binary Variables                              #
#=============================================================================# 
#--------------------------------------------------#
# Discretize generator variables (slack variables) #
#--------------------------------------------------#
M_G = [int(np.floor(np.log2(g_max[i])))for i in range(S)]
c_g = []
for i in range(S):
    coeff_list = [2**j for j in range(M_G[i])]
    coeff_list.append(g_max[i] + 1 - 2**M_G[i])
    c_g.append(coeff_list)
#----------------------------------------------------------#
# Creating the binary variables required for the CQM model #
#----------------------------------------------------------#
# Transmission Lines
X = [Binary('X_{}_{}'.format(i+1, j+1)) 
     for i in range(S) for j in range(i+1,S)]

# Generators with slack variables
G = [Binary('g_{}_{}'.format(i+1, j+1)) 
     for i in range(len(c_g)) for j in range(4)]

# Group the transmission lines in list of list [[LinesofNode1],...]
x = [[X[0], X[1]],[X[0],X[2]],[X[1],X[2]]]

# Group generators
g = []
list_gen = []
for item in G:
    list_gen.append(item)
    if len(list_gen)==4: #?
        g.append(list_gen)
        list_gen = []  

#=============================================================================#
#                               Build the CQM model                           #
#=============================================================================# 
cqm = ConstrainedQuadraticModel()
# Objective Function
obj_weight_value = 1

# Investment Cost
investment_objective = obj_weight_value * quicksum(C_iv[i] * X[i] 
                                                   for i in range(S))

# Operational cost
operational_objective = obj_weight_value * quicksum(C_oc[i]*c_g[i][j]*g[i][j] 
                                                    for i in range(S) 
                                                    for j in range(4))

# Total Objective
objective = investment_objective + operational_objective

# Add the objective to the CQM
cqm.set_objective(objective)

def pop_index(index, size):
    """
    Remove an index from a list.
    """
    relaxed_list = [i for i in range(size)]
    relaxed_list.remove(index)
    return relaxed_list

# Single Demand Constraint at Node 2
i=1 
cross_index  = pop_index(i, 3)
cqm.add_constraint(quicksum(c_g[i][j]*g[i][j] 
            + (c_g[cross_index[0]][j]*g[cross_index[0]][j])*x[i][0] 
            + (c_g[cross_index[1]][j]*g[cross_index[1]][j])*x[i][1] 
            for j in range(4)) == D[i], label = 'Demand_{}'.format(i+1))


#=============================================================================#
#                  Select a dimod Solver and solve the problem                #
#=============================================================================# 
cqm_exactsolver = dimod.ExactCQMSolver()
results = cqm_exactsolver.sample_cqm(cqm)