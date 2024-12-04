import numpy as np
import matplotlib.pyplot as plt
import math

def I(x):
    return np.exp(-100 * (x - L / 2)**2)

# Parameters
L = 10.0         # Length of the porous silicon layer (m)
T_total = 1000000  # Total simulation time (s)
T = 300         # Temperature in Kelvin

Nx = 1000       # Number of spatial points
Nt = 100000      # Number of time steps
D_bulk = 6.1e-6 # Bulk diffusion coefficient of H2 in air (m^2/s)
phi = 0.5       # Porosity of the material (dimensionless)
tau = 2.0       # Tortuosity factor (dimensionless)

# Effective diffusion coefficient for porous silicon
D_eff = phi * D_bulk / tau

# Derived parameters
dx = L / Nx
dt = T_total / Nt
F = D_eff * dt / dx**2  # Stability factor

E_a = 2.4 * (1.602e-19)*(6.022 * 10**23)   # Activation energy in Joules (converted from eV)
A = 10**13               # Pre-exponential factor in s^-1
R = 8.314                # Universal gas constant in J/mol·K

# Calculate the rate constant k using the Arrhenius equation
k = A * math.exp(-E_a / (R * T))

# Stability check
if F > 0.5:
    raise ValueError(f"Stability condition violated: F = {F:.2f}, must be <= 0.5")

# Spatial and temporal grids
x = np.linspace(0, L, Nx + 1)
u_old = 1 - (1 / (1 + np.exp(-20 * (x - L / 2))))  # Initial condition
u_new = np.copy(u_old)

# Time-stepping loop
for n in range(Nt):
    #u_new = u_old  # Create a new array for updated values
    for i in range(1, Nx):  # Skip boundary points
        diffusion = D_eff * (u_old[i-1] - 2*u_old[i] + u_old[i+1]) / (dx**2)
        reaction = k * u_old[i]
        u_new[i] = u_old[i] + dt * (diffusion - reaction)
    
    # Boundary conditions: Fixed concentration at boundaries
    u_new[0] = u_new[0] + dt * ((D_eff * ((u_new[1] - u_new[0]) / (dx**2))) - (k * u_new[0]))
    u_new[-1] = 0  
    u_old = u_new

# Plot the results
plt.plot(x, u_new, label=f'Time={T_total:.2f}s')
plt.plot(x, 1 - (1 / (1 + np.exp(-20 * (x - L / 2)))), label='Initial Condition', linestyle='--')
plt.title('Hydrogen Diffusion in Porous Silicon')
plt.xlabel('Distance (m)')
plt.ylabel('Concentration')
plt.legend()
plt.grid()
plt.show()