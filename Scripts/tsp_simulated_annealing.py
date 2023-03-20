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
    institution = {DLR - Deutsches Zentrumfür Luft- 
                   und Raumfahrt},
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
    Annealing Schedule choosen
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
#                                  Plotting                                   #
#=============================================================================#
def set_size(width,fraction=1, subplots=(1, 1), ratio=0):
    """
    Source: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    
    Set figure dimensions to avoid scaling in LaTeX.

    param width: float or string
            Document width in points, or string of predined document type
    param fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    param subplots: array-like, optional
            The number of rows and columns of subplots.
    returns:
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    if ratio == 0:
        golden_ratio = (5**.5 - 1) / 2
    else:
        golden_ratio = ratio  

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

# Latex Font See Ref. https://matplotlib.org/stable/tutorials/text/usetex.html
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "Palantino",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}

plt.style.use('default')
plt.rcParams.update(tex_fonts)

width = 'thesis'
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=set_size(width, subplots=(1,2),
 ratio=1.3))
fig.tight_layout(pad=1.5)


# Set names and Labels
ax1.set_xlabel('x - Distance [m]')
ax1.set_ylabel('y - distance [m]')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Hamiltonian')

# Take all the instances except the first and last one (we do not repeat nodes)
for first, second in zip(random_coords[:-1], random_coords[1:]):
    # Connect Nodes
    ax1.plot([first.x, second.x], [first.y, second.y], 'r',alpha=0.3)
# Connect first and last nodes
ax1.plot([random_coords[0].x, random_coords[-1].x], [random_coords[0].y,
 random_coords[-1].y], 'r', alpha=0.3, 
 label='Random Path: {} [m]'.format(round(cost_list[0],2)))

    
# Plot the result after Simulated Annealing Process
for first, second in zip(coords[:-1], coords[1:]):
    ax1.plot([first.x, second.x], [first.y, second.y], 'b')
ax1.plot([coords[0].x, coords[-1].x], [coords[0].y, coords[-1].y], 'b', 
label='Optimized Path: {} [m]'.format(round(cost_list[-1],2)))
# Represent each instances coordinates (nodes): Red Dots
for c in coords:
    ax1.plot(c.x, c.y, 'ko')

ax1.legend()

for c in coords:
    ax2.plot(c.x, c.y, 'ro')

# Gradient in scatter Function
ax2.set_ylim([0.9*min(cost_list), 1.1*max(cost_list[1:])])
im2 = ax2.scatter(steps_list, cost_list, s=5**2, c=T_list, cmap='plasma')
fig.colorbar(im2, ax=ax2).set_label('Temperature [K]')
plt.savefig('TSP_SA.pdf',  bbox_inches='tight')  
plt.show()