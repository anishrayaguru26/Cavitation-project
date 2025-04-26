import numpy as np
import matplotlib.pyplot as plt

# Constants for liquid sodium
L = 8.22e5  # Latent heat of vaporization for sodium (J/kg)
k = 71  # Thermal conductivity of liquid sodium (W/m·K)
p_w = 850  # Density of liquid sodium (kg/m³)
mu = 2.29e-4  # Dynamic viscosity of liquid sodium (Pa·s)
T_b = 700  # Base temperature for liquid sodium (K)
R_0 = 4e-6  # Initial bubble radius (m)
T_s_initial = T_b + 50  # Initial surface temperature of the bubble (K)
time_max = 1e-1  # Maximum simulation time (s)

# Initial conditions
R = R_0  # Initial bubble radius
T_s = T_s_initial  # Initial temperature at the bubble surface
dR_dt = 0  # Initial rate of radius change

# Time parameters - using logarithmic time steps for better resolution
times = np.logspace(-6, -1, 1000)  # From 1μs to 0.1s
dt = np.diff(times)[0]  # Initial time step

# Arrays to store values over time
R_values = []
T_s_values = []
dR_dt_values = []

# Vapor pressure function for sodium (Antoine-like equation)
def p_v(T):
    return np.exp(24.494 - 34100/T)  # Returns vapor pressure in Pa

# Loop over time steps to simulate bubble growth
for i, t in enumerate(times):
    if i > 0:
        dt = times[i] - times[i-1]
    
    # Calculate the temperature gradient at the surface
    # Using improved thermal boundary layer estimate for liquid metals
    T_gradient = (T_s - T_b) / (R * np.sqrt(k * t / p_w))

    # Rate of change of radius (from thermal balance)
    dR_dt = (k * T_gradient) / (p_w * L)

    # Update radius and temperature (with improved model for liquid metals)
    R += dR_dt * dt
    T_s = T_b + (L * p_w * dR_dt * R**2) / (k * R**2)
    
    # Ensure temperature stays within physical bounds
    T_s = max(min(T_s, T_b * 1.5), T_b * 0.5)

    # Store values for plotting
    R_values.append(R)
    T_s_values.append(T_s)
    dR_dt_values.append(dR_dt)

# Plot results
plt.figure(figsize=(10, 8))

# Plot bubble radius vs. time (log scale)
plt.subplot(2, 1, 1)
plt.semilogx(times, R_values, label="Bubble Radius")
plt.xlabel("Time (s)")
plt.ylabel("Radius (m)")
plt.title("Bubble Radius Evolution")
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()

# Plot surface temperature vs. time (log scale)
plt.subplot(2, 1, 2)
plt.semilogx(times, T_s_values, label="Surface Temperature", color="orange")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (K)")
plt.title("Surface Temperature Evolution")
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()

plt.tight_layout()
plt.show()
