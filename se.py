import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Physical properties for sodium
k = 71.7      # thermal conductivity (W/mK)
L = 2.6e6     # latent heat (J/kg)
rho_v = 0.5   # vapor density (kg/m³)
D = 6.4e-5    # thermal diffusivity (m²/s)
rho = 927     # liquid density (kg/m³)
p_inf = 1.0e5 # ambient pressure (Pa)
sigma = 0.2   # surface tension (N/m)

# Data from Table 1 in the paper
# Format: Tb (K), Delta_T (K), R0 (m), p_v (Pa), Jakob number
conditions = [
    (1083.6, 340.1, 2.5e-5, 1.253e5, 565.7),  # Case 1: very high superheat
    (1154.6, 278.9, 2.5e-5, 3.212e4, 227.7),  # Case 2: high superheat
    (1154.6, 133.1, 1e-4, 9.899e4, 108.7),    # Case 3: moderate superheat
    (1154.6, 22.1, 1e-3, 1.088e5, 18.04),     # Case 4: low superheat
    (1235.2, 90.1, 1e-4, 2.923e5, 38.08),     # Case 5: moderate superheat
    (1345.9, 14.7, 4e-4, 5.619e5, 2.979),     # Case 6: very low superheat
    (1390.2, 4.66, 1e-3, 2.936e4, 0.7331)     # Case 7: extremely low superheat
]

# Rather than solving the full integro-differential equations, 
# we use the asymptotic solutions derived in the paper

# Define functions for inertial and thermal growth velocities
def inertial_velocity(p_v, p_inf, rho):
    return np.sqrt(2/3 * (p_v - p_inf) / rho)

def thermal_velocity(k, L, rho_v, Delta_T, D, t):
    return (3/np.pi)**0.5 * k / (L * rho_v) * Delta_T / np.sqrt(D * t)

# Time array (logarithmic scale)
t = np.logspace(-7, -1, 500)  # seconds, from 10^-7 to 10^-1

plt.figure(figsize=(12, 8))

for i, (Tb, Delta_T, R0, p_v, Jakob) in enumerate(conditions):
    v_inertial = inertial_velocity(p_v, p_inf, rho) * np.ones_like(t)
    v_thermal = thermal_velocity(k, L, rho_v, Delta_T, D, t)
    
    # Plot inertial and thermal velocities
    plt.plot(t, v_inertial, label=f'Inertial growth velocity case {i+1}')
    plt.plot(t, v_thermal, label=f'Thermal growth velocity case {i+1}')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time (s)')
plt.ylabel('Growth velocity (m/s)')
plt.title('Growth rates of vapour bubbles in superheated sodium')
plt.legend(loc='upper right', fontsize='small')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
