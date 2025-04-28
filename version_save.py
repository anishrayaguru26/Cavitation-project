import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from tqdm import tqdm  # Import tqdm for progress bar

# Physical constants for water
rho_l = 958.4        # liquid density (kg/m³)
c_l = 4216           # specific heat capacity of liquid (J/kg/K)
lambda_l = 0.68      # thermal conductivity of liquid (W/m/K)
L = 2.257e6          # latent heat of vaporization (J/kg)
R_gas = 461.5        # specific gas constant for water vapor (J/kg/K)
sigma = 0.0589       # surface tension (N/m)
eta_l = 0   #2.82e-4      # dynamic viscosity (Pa.s)
p_inf = 1.013e5      # ambient pressure (Pa)
T_sat = 373.15       # saturation temperature at 1 atm (K)

# Specific heat at constant volume for vapor (approximate for water vapor)
c_v = 1418  # J/kg/K

# Initial bubble radius - will be set based on Delta_T
R_dot0 = 0.0 # initial bubble wall velocity

# Initial bubble radii for different superheating values based on the graph
R0_dict = {
    30: 3e-6,    # 3 micrometers for ΔT = 10K
    50: 3e-7,    # 300 nanometers for ΔT = 50K
    100: 8e-8    # 80 nanometers for ΔT = 100K
}

# Initial vapor density
rho_v0 = p_inf / (R_gas * T_sat)  # assume initial pressure ~ ambient pressure

# Time span
t_span = (1e-9, 1)
points = 5000
t_eval = np.logspace(np.log10(t_span[0]), np.log10(t_span[1]), points)

# Different initial superheating values (K)
Delta_T_list = R0_dict.keys()
Delta_T_list = sorted(Delta_T_list)  # Sort the keys for consistent plotting
colors = ['blue', 'orange', 'green']

temperature_K = np.array([
    273.15, 275, 280, 285, 290, 295, 300, 305, 310, 315, 320,
    325, 330, 335, 340, 345, 350, 355, 360, 365, 370, 373.15,
    375, 380, 385, 390, 400, 410, 420, 430, 440, 450, 460, 470,
    480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590,
    600, 610, 620, 625, 630, 635, 640, 645, 647.3
])

# Latent heat in kJ/kg
latent_heat_kJ_per_kg = np.array([
    2502, 2497, 2485, 2473, 2461, 2449, 2438, 2426, 2414, 2402, 2390,
    2378, 2366, 2354, 2342, 2329, 2317, 2304, 2291, 2278, 2265, 2257,
    2252, 2239, 2225, 2212, 2183, 2153, 2123, 2091, 2059, 2024, 1989,
    1951, 1912, 1870, 1825, 1779, 1730, 1679, 1622, 1564, 1499, 1429,
    1353, 1274, 1176, 1068, 941, 858, 781, 683, 560, 361, 0
])

# Create interpolation function
latent_heat_interp = interp1d(
    temperature_K, latent_heat_kJ_per_kg * 1000,  # Convert to J/kg
    kind='linear', bounds_error=False, fill_value=(latent_heat_kJ_per_kg[0]*1000, 0)
)


def latent_heat_water_interp(T_celsius):
    """
    Interpolated latent heat of vaporization for water based on temperature (°C).
    """
    T_kelvin = T_celsius + 273.15
    return latent_heat_interp(T_kelvin)

# Define the system of ODEs
def bubble_odes(t, y, T_inf):
    R, R_dot, rho_v, T_s = y

    if R <= 0:
        return [0, 0, 0, 0]

    # Temperature gradient at interface (assuming quasi-steady conduction)
    dTdr = (T_inf - T_s) / R
     
    L = latent_heat_water_interp(T_s - 273.15)  # Latent heat at interface temperature
    #print(L,T_s)
    # Mass flux at the interface
    j = lambda_l * dTdr / L

    # Liquid velocity at interface
    v_lR = R_dot - j / rho_l

    # Vapor pressure from ideal gas law
    #print(vapor_pressure_water(100))
    #p_v = vapor_pressure_water(T_s - 273.15) * 1e5  # Convert bars to Pa
    p_v = rho_v * R_gas * T_s  # Ideal gas law for vapor


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

# Use tqdm to create a progress bar for the superheating values
for Delta_T, color in tqdm(list(zip(Delta_T_list, colors)), desc="Processing superheating values"):
    T_inf = T_sat + Delta_T  # set liquid far-field temperature

    # Initial surface temperature assumed very close to T_sat
    T_s0 = T_sat #+ Delta_T

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

for Delta_T, color in tqdm(list(zip(Delta_T_list, colors)), desc="Processing velocity plots"):
    T_inf = T_sat + Delta_T  # set liquid far-field temperature
    
    # Initial surface temperature assumed very close to T_sat
    T_s0 = T_sat #+ Delta_T
    
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

# Create a new figure for temperature plot
plt.figure(figsize=(10,8))

for Delta_T, color in tqdm(list(zip(Delta_T_list, colors)), desc="Processing temperature plots"):
    T_inf = T_sat + Delta_T  # set liquid far-field temperature
    
    # Initial surface temperature assumed very close to T_sat
    T_s0 = T_sat #+ Delta_T
    
    # Initial conditions: [R, R_dot, rho_v, T_s]
    y0 = [R0_dict[Delta_T], R_dot0, rho_v0, T_s0]
    
    # Solve
    sol = solve_ivp(lambda t, y: bubble_odes(t, y, T_inf), t_span, y0, t_eval=t_eval, method='Radau', rtol=1e-6, atol=1e-9)
    
    # Extract results
    times = sol.t
    Temperatures = sol.y[3]  # T_s values
    
    # Plot T_s vs. Time
    plt.plot(times, Temperatures, label=rf'$\Delta T_i$ = {Delta_T} K', color=color)

# Final plot settings for temperature plot
plt.xscale('log')
plt.xlabel('Time (s)')
plt.ylabel('Bubble surface temperature (K)')
plt.title('Bubble Surface Temperature in Superheated Liquid')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.tight_layout()

# Save the temperature plot
plt.savefig('T_vs_t.png', dpi=300)

# Display the plots
plt.show()
