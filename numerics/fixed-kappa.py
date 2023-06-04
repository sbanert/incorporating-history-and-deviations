import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from matplotlib import rc

# Define the resolvent of A
def RA(y,γ):
    A = np.array([[0, -1], [1, 0]])
    return np.linalg.solve(np.eye(2) + γ*A, y)

# Algorithm parameters
γ = 0.1 # Step size
β = 0.001
N = 100000  # number of iterations
    
# Define the lambda sequence
def λ(n):
    return 1



# Define a list of κ values
κ_values = np.arange(-0.9,0.9+0.1,0.1) #[0, 0.2, 0.4, 0.6, 0.8, 1]

# Create a dictionary to store results
results = {}
run = 0 # Current run
iterations = []

for κ in κ_values:

    # Initialize the algorithm
    y = np.array([3,3]) # Initial guess
    p_prev = y # Prevous initial guess p
    x_prev = y
    y_prev = y
    x = y
    p = y
    u = np.array([0,0])

    # Store the updates
    y_values = [y]
    p_values = [p]
    
    # Initialization
    n = 0

    # Run the algorithm
    while n < N and np.linalg.norm(p) >= 1e-6:
        
        y = x+(1-λ(0)/λ(n))*(y_prev-x)+u
        
        p = RA(y,γ)

        x = x + λ(n)*(p-y) + (λ(n)-λ(0))*(y_prev-p_prev)

        u = κ*(4-γ*β-2*λ(0))/2*(p-x_prev+(1-λ(0)/λ(n))*(x_prev-p_prev) - (2-γ*β-2*λ(0))/(4-γ*β-2*λ(0))*u)

        # Move to the next iteration
        y_prev = y
        x_prev = x
        p_prev = p
        y_values.append(y)
        p_values.append(p)
        n += 1

    print(y)

    # Print the final y and p values
    print(f'After {n} iterations, y = {y}')
    print(f'After {n} iterations, p = {p}')

    # Store results
    results[run] = {'y': y_values, 'p': p_values}
    iterations.append(n)

    run += 1


# Plotting
# Plot trajectory
colors = cm.rainbow(np.linspace(0, 1, len(results)))

noPoints = 100
plt.figure(figsize=(6, 4.5))

for (_, color), (_, result) in zip(enumerate(colors), results.items()):
    y_values = result['y']
    p_values = result['p']
    plt.scatter([p[0] for p in p_values[:noPoints]], [p[1] for p in p_values[:noPoints]], color=color)

# Enable LaTeX 
rc('text', usetex=True)

plt.title(r'Algorithm trajectories')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
plt.show(block=False)

# Plot distance to solution
noPoints = 5000
plt.figure(figsize=(6, 4.5))

for (_, color), (_, result) in zip(enumerate(colors), results.items()):
    p_values = result['p']
    p_norm_values =  [np.sqrt(p[0]**2+p[1]**2) for p in p_values]
    plt.plot(p_norm_values[:noPoints], color=color)
    plt.yscale('log')

plt.title(r'Distance to solution')
plt.xlabel(r'Iteration')
plt.ylabel(r'$\|p_n-x^\star\|$')
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
plt.xlim(0,5000)
plt.ylim(1e-6,10)
plt.show(block=False)


