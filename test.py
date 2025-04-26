import numpy as np
import matplotlib.pyplot as plt

# Constants
L = 2260e3  # Latent heat of vaporization for water (J/kg)
k = 0.606  # Thermal conductivity of water at 100°C (W/m·K)
p_w = 997  # Density of water at 100°C (kg/m^3)
mu = 1e-3  # Dynamic viscosity of water at 100°C (Pa·s)
T_b = 373.15  # Boiling temperature of water (K)
R_0 = 1e-6  # Initial bubble radius (m)
T_s_initial = T_b + 20  # Initial surface temperature of the bubble (K)
time_max = 1e-3  # Maximum simulation time (s)

# Initial conditions
R = R_0  # Initial bubble radius
T_s = T_s_initial  # Initial temperature at the bubble surface
dR_dt = 0  # Initial rate of radius change

# Time parameters
dt = 1e-7  # Time step (s)
times = np.arange(0, time_max, dt)

# Arrays to store values over time
R_values = []
T_s_values = []
dR_dt_values = []

# Loop over time steps to simulate bubble growth
for t in times:
    # Calculate the temperature gradient at the surface
    # Simplified assumption: use a linear temperature gradient for thermal diffusion
    T_gradient = (T_s - T_b) / (R * np.sqrt(k * t))  # Using an estimate of the boundary layer

    # Rate of change of radius (from thermal balance)
    dR_dt = (k * T_gradient) / (p_w * L)

    # Update radius and temperature (simplified model)
    R += dR_dt * dt
    T_s = T_b + (L * p_w * dR_dt * R**2) / (k * R**2)

    # Store values for plotting
    R_values.append(R)
    T_s_values.append(T_s)
    dR_dt_values.append(dR_dt)

# Plot results
plt.figure(figsize=(12, 6))

# Plot bubble radius vs. time
plt.subplot(2, 1, 1)
plt.plot(times, R_values, label="Bubble Radius (m)")
plt.xlabel("Time (s)")
plt.ylabel("Radius (m)")
plt.title("Bubble Radius vs. Time")
plt.grid(True)

# Plot surface temperature vs. time
plt.subplot(2, 1, 2)
plt.plot(times, T_s_values, label="Surface Temperature (K)", color="orange")
plt.xlabel("Time (s)")
plt.ylabel("Surface Temperature (K)")
plt.title("Surface Temperature vs. Time")
plt.grid(True)

plt.tight_layout()
plt.show()
