import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Physical constants for water
rho_l = 958.4        # liquid density (kg/m³)
c_l = 4216           # specific heat capacity of liquid (J/kg/K)
lambda_l = 0.68      # thermal conductivity of liquid (W/m/K)
L = 2.257e6          # latent heat of vaporization (J/kg)
R_gas = 461.5        # specific gas constant for water vapor (J/kg/K)
sigma = 0.0589       # surface tension (N/m)
eta_l = 0#2.82e-4      # dynamic viscosity (Pa.s)
p_inf = 1.013e5      # ambient pressure (Pa)
T_sat = 373.15       # saturation temperature at 1 atm (K)

# Specific heat at constant volume for vapor (approximate for water vapor)
c_v = 1418  # J/kg/K

# Initial bubble radius - will be set based on Delta_T
R_dot0 = 0.0 # initial bubble wall velocity

# Initial bubble radii for different superheating values based on the graph
R0_dict = {
    20: 3e-6,    # 3 micrometers for ΔT = 10K
    50: 3e-7,    # 300 nanometers for ΔT = 50K
    100: 8e-8    # 80 nanometers for ΔT = 100K
}

# Initial vapor density
rho_v0 = p_inf / (R_gas * T_sat)  # assume initial pressure ~ ambient pressure

# Time span
t_span = (1e-9, 1e-4)
points = 300000
t_eval = np.logspace(np.log10(t_span[0]), np.log10(t_span[1]), points)

# Different initial superheating values (K)
Delta_T_list = [20, 50, 100]
colors = ['blue', 'orange', 'green']

# Define the system of ODEs
def bubble_odes(t, y, T_inf):
    R, R_dot, rho_v, T_s = y

    if R <= 0:
        return [0, 0, 0, 0]

    # Temperature gradient at interface (assuming quasi-steady conduction)
    dTdr = (T_inf - T_s) / R

    # Mass flux at the interface
    j = lambda_l * dTdr / L

    # Liquid velocity at interface
    v_lR = R_dot - j / rho_l

    # Vapor pressure from ideal gas law
    p_v = rho_v * R_gas * T_s

    # Interface pressure balance (includes surface tension and viscosity terms)
    p_lR = p_v + j * v_lR - 2 * sigma / R - 4 * eta_l * v_lR / R

    # Rayleigh-Plesset type equation (modified)
    dR_dt = R_dot
    R_ddot = (p_lR - p_inf) / (rho_l * R) + (v_lR**2) / (2 * R) - 2 * v_lR * R_dot / R

    # Mass conservation inside bubble
    drho_v_dt = (3 * (j - rho_v * v_lR)) / R

    # Energy conservation at the interface (liquid side energy loss)
    #Grad_dist = 10*R
    alpha = lambda_l / (rho_l * c_l)  # thermal diffusivity of liquid
    beta = 1000 #thermal enhancement factor (assumed)
    Grad_dist = beta*np.sqrt(np.pi * alpha * t)  # characteristic length scale for diffusion
    dT_s_dt = (lambda_l / (rho_l * c_l)) * ((T_inf - T_s) / (Grad_dist**2)) - ((v_lR) * ((T_inf - T_s)/Grad_dist))

    return [dR_dt, R_ddot, drho_v_dt, dT_s_dt]

# Plotting
plt.figure(figsize=(10,8))

for Delta_T, color in zip(Delta_T_list, colors):
    T_inf = T_sat + Delta_T  # set liquid far-field temperature

    # Initial surface temperature assumed very close to T_sat
    T_s0 = T_sat

    # Initial conditions: [R, R_dot, rho_v, T_s]
    y0 = [R0_dict[Delta_T], R_dot0, rho_v0, T_s0]

    # Solve
    sol = solve_ivp(lambda t, y: bubble_odes(t, y, T_inf), t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-9)

    # Extract results
    times = sol.t
    Radii = sol.y[0]
    Velocities = sol.y[1]  # dR/dt values

    # Plot Radius vs. Time
    plt.plot(times, Radii, label=rf'$\Delta T_i$ = {Delta_T} K', color=color)

# Final plot settings for radius plot
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time (s)')
plt.ylabel('Bubble radius (m)')
plt.title('Bubble Growth in Superheated Liquid (with Mass & Energy Transfer)')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.tight_layout()

# Save the radius plot
plt.savefig('bubble_radius_vs_time.png', dpi=300)

# Create a new figure for velocity plot
plt.figure(figsize=(10,8))

for Delta_T, color in zip(Delta_T_list, colors):
    T_inf = T_sat + Delta_T  # set liquid far-field temperature
    
    # Initial surface temperature assumed very close to T_sat
    T_s0 = T_sat
    
    # Initial conditions: [R, R_dot, rho_v, T_s]
    y0 = [R0_dict[Delta_T], R_dot0, rho_v0, T_s0]
    
    # Solve
    sol = solve_ivp(lambda t, y: bubble_odes(t, y, T_inf), t_span, y0, t_eval=t_eval, method='Radau', rtol=1e-6, atol=1e-9)
    
    # Extract results
    times = sol.t
    Velocities = sol.y[1]  # dR/dt values
    
    # Plot dR/dt vs. Time
    plt.plot(times, Velocities, label=rf'$\Delta T_i$ = {Delta_T} K', color=color)

# Final plot settings for velocity plot
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time (s)')
plt.ylabel('Bubble wall velocity dR/dt (m/s)')
plt.title('Bubble Wall Velocity in Superheated Liquid (with Mass & Energy Transfer)')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.tight_layout()

# Save the velocity plot
plt.savefig('bubble_velocity_vs_time.png', dpi=300)

# Display the plots
plt.show()
