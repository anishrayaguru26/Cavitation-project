import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm

# Physical constants for water
rho_l = 958.4        # liquid density (kg/m^3)
c_l = 4216           # liquid specific heat (J/kg/K)
lambda_l = 0.68      # thermal conductivity (W/m/K)
eta_l = 2.82e-4      # viscosity (Pa.s)
sigma = 0.0589       # surface tension (N/m)
L = 2.257e6          # latent heat (J/kg)
R_gas = 461.5        # specific gas constant for water vapor (J/kg/K)

# External conditions
T_inf = 373.15 + 50   # surrounding liquid temperature (K), e.g., superheat 50K
T_sat = 373.15        # saturation temperature at 1 atm (K)
p_inf = 1.013e5       # ambient pressure (Pa)
p_sat = 2.33e3        # saturation pressure at T_sat (Pa)

# Initial bubble radius
R0 = 1e-6  # 1 micron initial radius

# Initial vapor density
#p_sat = p_inf * np.exp((L/R_gas)*(1/T_sat - 1/T_inf))  # Clausius-Clapeyron
rho_v0 = p_sat / (R_gas * T_inf)  # ideal gas law

# Functions
def temperature_gradient(R):
    """Approximate steady temperature gradient at bubble interface"""
    return (T_inf - T_sat) / R

def j_evap(R):
    """Mass flux due to evaporation (kg/m^2/s)"""
    return lambda_l * temperature_gradient(R) / L

def system(t, y):
    """System of ODEs: [R, rho_v]"""
    R, rho_v = y

    if R <= 0:
        return [0.0, 0.0]

    # Mass flux at the interface
    j = j_evap(R)

    # Approximate V_LR
    V_LR = (j / rho_l)

    # Calculate dR/dt from modified Rayleigh-Plesset equation (simplified)
    pressure_difference = p_sat - p_inf
    dR_dt = np.sqrt(max(2 * pressure_difference / rho_l, 0))

    # Calculate d(rho_v)/dt from mass conservation
    d_rho_v_dt = (3 / R) * (j - rho_v * (dR_dt / R))

    return [dR_dt, d_rho_v_dt]

# Initial conditions
y0 = [R0, rho_v0]

# Time span
t_span = (1e-9, 1e-4)
t_eval = np.logspace(-9, -4, 500)

# Create a progress bar for integration
pbar = tqdm(total=100, desc="Solving ODE system", unit="%", ncols=70)
last_t = [t_span[0]]  # Store last reported time in a mutable container

# Callback function to update progress bar during integration
def update_progress(t, y):
    # Calculate progress as percentage of time span completed
    progress = 100 * (t - t_span[0]) / (t_span[1] - t_span[0])
    # Update progress bar with increment
    increment = max(0, progress - (100 * (last_t[0] - t_span[0]) / (t_span[1] - t_span[0])))
    pbar.update(increment)
    last_t[0] = t
    return False  # Return False to continue integration

# Solve the system
sol = solve_ivp(system, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-9,
                events=update_progress)

# Close the progress bar
pbar.close()

# Extract solution
R_sol = sol.y[0]
t_sol = sol.t

# Plot
plt.figure(figsize=(8,6))
plt.loglog(t_sol, R_sol, label="Direct numerical", color='red')
plt.xlabel("Time (s)")
plt.ylabel("Bubble radius (m)")
plt.grid(True, which='both', ls='--')
plt.legend()
plt.title("Bubble Growth in Superheated Liquid (Direct Numerical)")
plt.tight_layout()
plt.show()
