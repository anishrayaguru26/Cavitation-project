import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm

# Temperature-dependent property functions
@jit(nopython=True)
def get_latent_heat(T):
    """Calculate latent heat of vaporization (J/kg) for sodium"""
    Tc = 2503.7  # Critical temperature (K)
    L0 = 4.1e6   # Reference latent heat
    n = 0.38     # Correlation exponent
    tr = T/Tc
    if tr >= 1.0:
        return 0.0
    return L0 * (1 - tr)**n

@jit(nopython=True)
def get_liquid_density(T):
    """Calculate liquid density (kg/m³) for sodium"""
    rho_0 = 927.0  # Reference density
    beta = 2.5e-4  # Thermal expansion coefficient
    return rho_0 * (1.0 - beta * (T - T_INF))

@jit(nopython=True)
def get_surface_tension(T):
    """Calculate surface tension (N/m) for sodium"""
    sigma_0 = 0.2  # Reference surface tension
    dgamma = -1e-4 # Temperature coefficient
    return max(0.0, sigma_0 + dgamma * (T - T_INF))

@jit(nopython=True)
def calculate_vapor_pressure(T):
    """Calculate vapor pressure (Pa) using Antoine equation for sodium"""
    ln_P = 11.9463 - 12633.73/T - 0.4672 * np.log(T)
    return np.exp(ln_P) * 1e6

@jit(nopython=True)
def calculate_vapor_density(T):
    """Calculate vapor density (kg/m³) for sodium using ideal gas approximation"""
    M = 0.023  # kg/mol (sodium molar mass)
    R = 8.314  # J/(mol·K)
    P = calculate_vapor_pressure(T)
    return (M * P)/(R * T)

@jit(nopython=True)
def safe_divide(a, b, default=0.0):
    """Safe division with error handling"""
    try:
        if abs(b) < 1e-15:
            return default
        result = a / b
        if np.isfinite(result):
            return result
        return default
    except:
        return default

@jit(nopython=True)
def clip_value(x, min_val, max_val):
    """Numba-compatible value clipping"""
    return max(min_val, min(x, max_val))

@jit(nopython=True)
def system_derivatives(R, V, T_s, params):
    """Calculate derivatives for both R and V"""
    dRdt = V
    _, dVdt = rayleigh_plesset_optimized(R, V, 0.0, T_s, params)
    return dRdt, dVdt

@jit(nopython=True)
def energy_check(R, V, T_s, params):
    """Check energy conservation"""
    rho_l, k, D, L, p_inf, sigma, T_inf, rho_v = params
    
    # Kinetic energy
    E_k = 2 * np.pi * rho_l * R**3 * V**2
    
    # Surface energy
    E_s = 4 * np.pi * R**2 * sigma
    
    # Thermal energy
    E_t = 4/3 * np.pi * R**3 * rho_v * L * (T_s - T_inf) / T_s
    
    return E_k + E_s + E_t

@jit(nopython=True)
def rk4_step(R, V, T_s, dt, params):
    """
    4th order Runge-Kutta with energy conservation
    """
    # Store initial energy
    E_init = energy_check(R, V, T_s, params)
    
    # Standard RK4 steps
    k1_R, k1_V = system_derivatives(R, V, T_s, params)
    
    R2 = R + 0.5 * dt * k1_R
    V2 = V + 0.5 * dt * k1_V
    k2_R, k2_V = system_derivatives(R2, V2, T_s, params)
    
    R3 = R + 0.5 * dt * k2_R
    V3 = V + 0.5 * dt * k2_V
    k3_R, k3_V = system_derivatives(R3, V3, T_s, params)
    
    R4 = R + dt * k3_R
    V4 = V + dt * k3_V
    k4_R, k4_V = system_derivatives(R4, V4, T_s, params)
    
    # Update with energy check
    R_new = R + (dt/6.0) * (k1_R + 2*k2_R + 2*k3_R + k4_R)
    V_new = V + (dt/6.0) * (k1_V + 2*k2_V + 2*k3_V + k4_V)
    
    # Check energy conservation
    E_final = energy_check(R_new, V_new, T_s, params)
    
    # Adjust velocity to conserve energy if needed
    if abs(E_final - E_init) > 0.1 * abs(E_init):
        scaling = np.sqrt(abs(E_init/E_final))
        V_new *= scaling
    
    # Apply physical bounds
    R_new = max(R_new, 1e-12)
    V_new = clip_value(V_new, -1e6, 1e6)
    
    return R_new, V_new

@jit(nopython=True)
def calculate_derived_properties(T_s, params):
    """Calculate all temperature-dependent properties efficiently"""
    rho_l, k, D, L, p_inf, sigma, T_inf, _ = params
    
    # Calculate properties
    p_v = calculate_vapor_pressure(T_s)
    rho_v = calculate_vapor_density(T_s)
    L_val = get_latent_heat(T_s)
    sigma_val = get_surface_tension(T_s)
    rho_l_val = get_liquid_density(T_s)
    mu_val = 2.29e-4 * (T_s/1154.0)**0.5
    
    return p_v, rho_v, L_val, sigma_val, rho_l_val, mu_val

@jit(nopython=True)
def inertial_timescale(R, V):
    """Calculate inertial timescale"""
    return np.sqrt(abs(R / (V + 1e-10)))

@jit(nopython=True)
def thermal_timescale(R):
    """Calculate thermal diffusion timescale"""
    return R * R / D_L

@jit(nopython=True)
def phase_change_timescale(R, T_s):
    """Calculate phase change timescale"""
    return RHO_L * CP_L * R / (K_L * (T_s - T_INF + 1e-10))

@jit(nopython=True)
def get_adaptive_dt(t, R, V, T_s):
    """
    Adaptive timestep based on multiple physical timescales
    """
    tau_i = inertial_timescale(R, V)
    tau_th = thermal_timescale(R)
    tau_pc = phase_change_timescale(R, T_s)
    
    # Use minimum timescale with safety factor
    dt = 0.1 * min(tau_i, tau_th, tau_pc)
    return clip_value(dt, 1e-15, 1e-3)

@jit(nopython=True)
def rayleigh_plesset_optimized(R, V, t, T_s, params):
    """
    Rayleigh-Plesset equation with improved thermal effects
    """
    rho_l, k, D, L, p_inf, sigma, T_inf, rho_v = params
    
    # Get temperature-dependent properties
    p_v = calculate_vapor_pressure(T_s)
    rho_v = calculate_vapor_density(T_s)
    L_val = get_latent_heat(T_s)
    sigma_val = get_surface_tension(T_s)
    rho_l_val = get_liquid_density(T_s)
    mu = 2.29e-4 * (T_s/1154.0)**0.5
    
    # Pressure terms
    p_vapor = p_v
    p_thermal = rho_v * L_val * (T_s - T_inf) / T_s
    p_surface = 2 * sigma_val / R
    p_viscous = 4 * mu * V / R
    
    # Total pressure difference
    delta_p = p_vapor + p_thermal - p_inf - p_surface - p_viscous
    
    # Acceleration terms
    dVdt = delta_p/(rho_l_val * R) - 1.5 * (V * V)/R
    
    return V, dVdt

@jit(nopython=True)
def integrate_bubble_dynamics(t, R, V, T_s, dt, params):
    """
    4th order Runge-Kutta integration with stability controls
    """
    # RK4 for coupled R-V system
    k1_V, k1_A = rayleigh_plesset_optimized(R, V, t, T_s, params)
    
    R2 = R + 0.5 * dt * k1_V
    V2 = V + 0.5 * dt * k1_A
    k2_V, k2_A = rayleigh_plesset_optimized(R2, V2, t + 0.5*dt, T_s, params)
    
    R3 = R + 0.5 * dt * k2_V
    V3 = V + 0.5 * dt * k2_A
    k3_V, k3_A = rayleigh_plesset_optimized(R3, V3, t + 0.5*dt, T_s, params)
    
    R4 = R + dt * k3_V
    V4 = V + dt * k3_A
    k4_V, k4_A = rayleigh_plesset_optimized(R4, V4, t + dt, T_s, params)
    
    # Update with stability checks
    R_new = R + (dt/6) * (k1_V + 2*k2_V + 2*k3_V + k4_V)
    V_new = V + (dt/6) * (k1_A + 2*k2_A + 2*k3_A + k4_A)
    
    # Apply physical constraints
    R_new = max(R_crit/2, min(R_new, 1e-2))
    V_new = clip_value(V_new, -100.0, 100.0)
    
    return R_new, V_new

@jit(nopython=True)
def calculate_timestep(R, V, T_s, t):
    """
    Adaptive timestep calculation considering all relevant timescales
    """
    # Thermal timescale
    tau_th = R * R / D_L
    
    # Inertial timescale
    tau_i = R / (abs(V) + 1e-10)
    
    # Interface timescale
    tau_int = np.sqrt(RHO_L * R**3 / (2 * SIGMA))
    
    # Use smallest timescale with safety factor
    dt = 0.1 * min(tau_th, tau_i, tau_int)
    return clip_value(dt, 1e-12, 1e-4)

def calculate_temperature_plesset_zwick(t, R_hist, dRdt_hist, t_hist, T_prev):
    """
    Temperature calculation using Plesset-Zwick theory
    """
    if len(t_hist) < 3:
        return T_LIQUID_INITIAL
        
    try:
        # Current state
        R_current = R_hist[-1]
        V_current = dRdt_hist[-1]
        dt = t - t_hist[-2]
        
        # Thermal boundary layer thickness
        delta = np.sqrt(D_L * t)
        
        # Heat flux through interface
        q_cond = K_L * (T_LIQUID_INITIAL - T_prev) / delta
        q_latent = L_VAP * calculate_vapor_density(T_prev) * V_current
        q_total = q_cond + q_latent
        
        # Temperature change from energy balance
        dT = -q_total * dt / (RHO_L * CP_L * delta)
        
        # Limit temperature change rate
        max_dT = 50.0  # Maximum temperature change per step
        dT = clip_value(dT, -max_dT, max_dT)
        
        # Update with physical bounds
        T_new = T_prev + dT
        return clip_value(T_new, T_MIN, T_LIQUID_INITIAL)
        
    except Exception as e:
        return T_prev

# ---------------------------------------------------------------------
# 1. Physical Constants and Initial Conditions
# ---------------------------------------------------------------------
# Update initial conditions
R0 = 2.5e-5  # m (from paper Case 1)
V0 = 0.0  # Start from rest
T_SUPERHEAT = 340.1  # K (Case 1)
T_INF = 1083.6  # K (Case 1)
T_LIQUID_INITIAL = T_INF + T_SUPERHEAT
P_INF = 1.253e5  # Pa (Case 1)

# Physical properties matching paper
K_L = 71.7  # W/(m·K)
CP_L = 1380.0  # J/(kg·K)
RHO_L = 927.0  # kg/m³
SIGMA = 0.2  # N/m
D_L = K_L / (RHO_L * CP_L)
L_VAP = 4.1e6  # J/kg

# Temperature bounds
T_MIN = T_INF - 50  # Lower bound
T_MAX = T_LIQUID_INITIAL + 50  # Upper bound

# Initial conditions
p_v_init = calculate_vapor_pressure(T_LIQUID_INITIAL)
rho_v_init = calculate_vapor_density(T_LIQUID_INITIAL)
delta_p = p_v_init - P_INF  # Initial pressure difference

# Critical radius from mechanical equilibrium
R_crit = 2 * SIGMA / max(delta_p, 1e3)  # Ensure positive value

# Time stepping parameters
t_start = 1e-8  # Initial time (s)
t_end = 1.0     # End time (s)
n_steps = 10000  # Number of time steps

# Calculate characteristic times
tau_thermal = R0 * R0 / D_L  # Thermal diffusion time
tau_inertial = np.sqrt(RHO_L * R0**3 / (2 * SIGMA))  # Inertial time
dt_init = min(tau_thermal, tau_inertial) / 100  # Initial time step

print(f"Initial pressure difference: {delta_p/1e5:.2f} bar")
print(f"Critical radius: {R_crit:.3e} m")
print(f"Initial radius: {R0:.3e} m")
print(f"Characteristic times:")
print(f"  Thermal: {tau_thermal:.3e} s")
print(f"  Inertial: {tau_inertial:.3e} s")

# Pre-allocate arrays with adaptive size
max_steps = int(1.5 * n_steps)  # Allow for extra steps
t_history = np.zeros(max_steps)
R_history = np.zeros(max_steps)
V_history = np.zeros(max_steps)
Ts_history = np.zeros(max_steps)

# Initialize first elements
t_history[0] = t_start
R_history[0] = R0
V_history[0] = V0
Ts_history[0] = T_LIQUID_INITIAL

# Initialize simulation variables
t = t_start
R = R0
V = V0
Ts = T_LIQUID_INITIAL
i = 0

# Initial parameter set
params = (RHO_L, K_L, D_L, L_VAP, P_INF, SIGMA, T_INF, rho_v_init)

print(f"\nStarting simulation:")
print(f"R0={R0:.3e} m")
print(f"T0={T_LIQUID_INITIAL:.1f} K")
print(f"P_inf={P_INF/1e5:.2f} bar")
print(f"Initial p_v={p_v_init/1e5:.3f} bar")

# Main time stepping loop with adaptive dt
with tqdm(total=n_steps, desc="Simulating") as pbar:
    while t < t_end and i < max_steps-1:
        # Get adaptive timestep
        dt = calculate_timestep(R, V, Ts, t)
        
        # Integrate bubble dynamics
        R_new, V_new = integrate_bubble_dynamics(t, R, V, Ts, dt, params)
        
        # Update temperature
        t_new = t + dt
        Ts_new = calculate_temperature_plesset_zwick(t_new, R_history[:i+1], V_history[:i+1], t_history[:i+1], Ts)
        
        # Store results
        i += 1
        t_history[i] = t_new
        R_history[i] = R_new
        V_history[i] = V_new
        Ts_history[i] = Ts_new
        
        # Update state
        t = t_new
        R = R_new
        V = V_new
        Ts = Ts_new
        
        # Update progress about every 5%
        if i % (n_steps//20) == 0:
            pbar.update(n_steps//20)
            print(f"\nt={t:.3e} s, R={R:.3e} m, V={V:.2f} m/s, Ts={Ts:.1f} K")

# Trim arrays to actual size
n_actual = i + 1
t_history = t_history[:n_actual]
R_history = R_history[:n_actual]
V_history = V_history[:n_actual]
Ts_history = Ts_history[:n_actual]

print("\nSimulation finished")

# ---------------------------------------------------------------------
# 7. Plotting with exact paper format
# ---------------------------------------------------------------------
# Convert units to match paper
t_plot = t_history  # seconds
R_plot = R_history * 100  # Convert m to cm
V_plot = V_history * 100  # Convert m/s to cm/s
T_plot = Ts_history  # Kelvin

# Create figure without seaborn dependency
plt.figure(figsize=(8, 13))

# Plot 1: Radius vs time (log-log)
plt.subplot(311)
plt.loglog(t_plot, R_plot, 'k-', linewidth=1.5)
plt.xlim(1e-8, 1)
plt.ylim(1e-4, 1e-1)
plt.ylabel('R (cm)')
plt.grid(True, which='both', alpha=0.2)
plt.grid(True, which='minor', linestyle=':', alpha=0.1)

# Plot 2: Temperature vs time (log-x, linear-y)
plt.subplot(312)
plt.semilogx(t_plot, T_plot, 'k-', linewidth=1.5)
plt.xlim(1e-7, 1)
plt.ylim(1100, 1500)
plt.ylabel('T$_s$ (K)')
plt.grid(True, which='both', alpha=0.2)
plt.grid(True, which='minor', linestyle=':', alpha=0.1)

# Plot 3: Velocity vs time (log-log)
plt.subplot(313)
plt.loglog(t_plot, np.abs(V_plot), 'k-', linewidth=1.5)
plt.xlim(1e-8, 1e-1)
plt.ylim(1e1, 1e4)
plt.xlabel('t (s)')
plt.ylabel('|dR/dt| (cm/s)')
plt.grid(True, which='both', alpha=0.2)
plt.grid(True, which='minor', linestyle=':', alpha=0.1)

plt.suptitle('Vapor Bubble Growth in Superheated Sodium\n' + 
             f'(T$_∞$ = {T_INF:.1f} K, ΔT = {T_SUPERHEAT:.1f} K)', y=0.95)
plt.tight_layout()
plt.savefig('bubble_dynamics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()