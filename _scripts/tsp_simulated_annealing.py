#=============================================================================#
# Traveling salesman problem by simulated annealing 
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
import copy
import math
import matplotlib.pyplot as plt
import numpy as np


#=============================================================================#
#                              Coordinate Class                               #
#=============================================================================#
class Coordinates:
    def __init__(self, x, y):
        """
        Store a given coordinates array x, y
        """
        self.x = x
        self.y = y

    @staticmethod
    def get_distance(a, b): 
        """
        Get distance from two nodes (currrent node and next node):  
        param a: vector (x,y) of node 'a'
        param b: vector (x,y) of node 'b'
        """
        return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

    @staticmethod
    def get_total_distance(coords):
        """
        Get distance of the path for the current configuration.
        param coords: numpy.array with the coordinates of all the nodes 
        arranged in the travelled order.
        """
        dist = 0   # Initialize the distance
        # Zip combine all coordinates x,y of nodes except the first and last 
        # node
        for first, second in zip(coords[:-1], coords[1:]):       
            dist += Coordinates.get_distance(first, second)
        # Distance from first node to last node
        dist += Coordinates.get_distance(coords[0], coords[-1])  
        return dist


#=============================================================================#
#            Fill up the coordinates and connecting nodes randomly            #
#=============================================================================#
coords = []
N = 50  # Number of nodes
for i in range(N):
    # Create a random list of instances (Nodes): Each instance with x = self.x 
    # and y = self.y coordinates
    coords.append(Coordinates(np.random.uniform(low=0.0, high=1),
     np.random.uniform(low=0.0, high=1)))
random_coords = copy.deepcopy(coords)


#=============================================================================#
#                        Simulated annealing algorithm                        #
#=============================================================================#
print("Starting annealing...")
#=============================================================================#
#                              Annealing Schedule                             #
#=============================================================================#
def annealing_schedule(T, factor, cost_init, number_of_swaps, coords,
 annealing_type=None):
    """
    Annealing Schedule
    """
    # Type of Schedule (more types can be added)
    if annealing_type is None or annealing_type == "linear":
        T *= factor
       
    for j in range(number_of_swaps): 
        # Exchange two coordinates and get a new neighbour solution
        # Get Index of two coordinates to swap
        r1, r2 = np.random.randint(0, len(coords), size=2)

        temp = coords[r1]
        coords[r1] = coords[r2]
        coords[r2] = temp

        # Get the cost of the new path
        cost1 = Coordinates.get_total_distance(coords)
        # Acceptance of new Candidate path
        # Check if neighbor is best so far
        cost_diff = cost1 - cost_init
        # if the new solution is better, accept it
        if cost_diff < 0:
            cost_init = cost1
        # if the new solution is not better, use Metropolis criterion
        else:
            if np.random.uniform() < math.exp(-cost_diff/T):
                cost_init = cost1
            else:
                temp = coords[r1]
                coords[r1] = coords[r2]
                coords[r2] = temp
    return [T, cost_init]


# List to store values
cost_list = []  
T_list = []
steps_list = [0]

cost0 = Coordinates.get_total_distance(coords) # Current Cost
cost_list.append(cost0)
# Number of swaps between nodes before decreasing the temperature
number_of_swaps = 500

# Iterations equiv. decrease temperature by "factor", 
# number_of_iterations (1000 times)
iteration = 0
number_of_iterations = 1000
# Every 10 steps we store the cost
steps = 10  

# Annealing parameters
T = 20       # Current Temperature
T_init = T   
T_end = 0.01
factor = (T_end/T_init)**(1/number_of_iterations)
T_list.append(T)

# Main Loop
while T > T_end:
    iteration += 1
    print("Temperature:", T, "[K]  ", 'cost = ', cost0,
     'Iteration = ', iteration)
    # Append Cost and temperature Every 100 iterations
    if (iteration % steps) == 0:
        steps_list.append(iteration)
        cost_list.append(cost0)
        T_list.append(T)
        if cost_list[-1] == cost_list[-2]:
            break 
    list_annealing_schedule = annealing_schedule(T, factor, cost0,
     number_of_swaps, coords, "linear")
    T = list_annealing_schedule[0]
    cost0 = list_annealing_schedule[1]

#=============================================================================#
# Plot with your preferred framework                                   
#=============================================================================#
