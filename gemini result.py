import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm

# Temperature-dependent property functions
@jit(nopython=True)
def calculate_vapor_pressure(T):
    """
    Calculate vapor pressure (Pa) for sodium using Antoine equation
    Valid for temperature range 700-2500K
    """
    A = 11.9463
    B = -12633.73
    C = -0.4672
    ln_p = A + B/T + C * np.log(T)
    return np.exp(ln_p) * 1e6  # Convert to Pa

@jit(nopython=True)
def calculate_vapor_density(T):
    """
    Calculate vapor density (kg/m³) for sodium using gas law with compressibility
    """
    M = 0.023  # kg/mol (sodium molar mass)
    R = 8.314  # J/(mol·K)
    p = calculate_vapor_pressure(T)
    Z = 1.0  # Compressibility factor (near ideal at these conditions)
    return (M * p)/(Z * R * T)

@jit(nopython=True)
def get_latent_heat(T):
    """
    Calculate latent heat of vaporization (J/kg) for sodium with T dependence
    """
    Tc = 2503.7  # Critical temperature (K)
    L0 = 4.1e6   # Reference latent heat at Tb
    tr = T/Tc
    if tr >= 1.0:
        return 0.0
    return L0 * (1 - tr)**0.38  # Temperature dependent form

@jit(nopython=True)
def get_liquid_density(T):
    """
    Calculate liquid density (kg/m³) for sodium with T dependence
    """
    rho_0 = 927.0  # Reference density at T_inf
    beta = 2.5e-4  # Thermal expansion coefficient
    dT = T - T_INF
    return rho_0 * (1.0 - beta * dT)

@jit(nopython=True)
def get_surface_tension(T):
    """
    Calculate surface tension (N/m) for sodium with T dependence
    """
    sigma_0 = 0.2  # Reference surface tension
    dsigma_dT = -1e-4  # Temperature coefficient
    dT = T - T_INF
    return max(0.0, sigma_0 + dsigma_dT * dT)

@jit(nopython=True)
def get_thermal_conductivity(T):
    """
    Calculate thermal conductivity (W/m·K) for sodium with T dependence
    """
    k_0 = 71.7  # Reference conductivity
    dk_dT = -0.05  # Temperature coefficient
    dT = T - T_INF
    return k_0 + dk_dT * dT

@jit(nopython=True)
def get_specific_heat(T):
    """
    Calculate specific heat (J/kg·K) for sodium with T dependence
    """
    cp_0 = 1380.0  # Reference specific heat
    dcp_dT = 0.1  # Temperature coefficient
    dT = T - T_INF
    return cp_0 + dcp_dT * dT

@jit(nopython=True)
def get_thermal_diffusivity(T):
    """
    Calculate thermal diffusivity (m²/s) for sodium
    """
    k = get_thermal_conductivity(T)
    rho = get_liquid_density(T)
    cp = get_specific_heat(T)
    return k/(rho * cp)

@jit(nopython=True)
def get_dynamic_viscosity(T):
    """
    Calculate dynamic viscosity (Pa·s) for sodium with T dependence
    """
    mu_0 = 2.29e-4  # Reference viscosity
    T_ref = 1154.0  # Reference temperature
    return mu_0 * (T/T_ref)**0.5

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
    mu_val = get_dynamic_viscosity(T_s)
    
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
    Rayleigh-Plesset equation with improved stability and physics
    """
    rho_l, k, D, L, p_inf, sigma, T_inf, rho_v = params
    
    # Get temperature-dependent properties
    p_v = calculate_vapor_pressure(T_s)
    rho_v = calculate_vapor_density(T_s)
    L_val = get_latent_heat(T_s)
    sigma_val = get_surface_tension(T_s)
    mu = get_dynamic_viscosity(T_s)
    
    # Enhanced superheat effect
    p_vapor = p_v * (1.0 + 0.05 * (T_s - T_inf)/T_inf)  # Increased temperature sensitivity
    
    # Improved thermal pressure term
    p_thermal = rho_v * L_val * (T_s - T_inf) / T_s  # Full thermal pressure contribution
    
    # Surface tension with temperature dependence
    p_surface = 2 * sigma_val / R
    
    # Enhanced viscous damping
    p_viscous = 4 * mu * V / R
    
    # Growth limiting based on modified Weber number
    We = rho_l * R * abs(V) * abs(V) / sigma_val  # Using absolute velocity
    f_We = 1.0 / (1.0 + 0.01 * We)  # Reduced growth limiting
    
    # Total pressure difference with growth enhancement for small radii
    r_ratio = R / R_crit
    growth_enhancement = np.exp(-0.1 * r_ratio)  # Enhanced growth for small bubbles
    
    delta_p = ((p_vapor + p_thermal - p_inf - p_surface) * f_We * (1.0 + growth_enhancement) 
               - p_viscous)
    
    # Modified acceleration term with improved stability
    dVdt = (delta_p / (rho_l * R)) - (3 * V * abs(V))/(2 * R)  # Using abs(V) for better damping
    
    return V, dVdt

def calculate_temperature_plesset_zwick(t, R_hist, dRdt_hist, t_hist, T_prev):
    """
    Temperature calculation with enhanced cooling
    """
    if len(t_hist) < 3:
        return T_LIQUID_INITIAL
        
    try:
        # Current state
        R_current = R_hist[-1]
        V_current = dRdt_hist[-1]
        dt = t - t_hist[-2]
        
        # Get temperature-dependent properties
        k = get_thermal_conductivity(T_prev)
        cp = get_specific_heat(T_prev)
        rho = get_liquid_density(T_prev)
        alpha = k / (rho * cp)
        
        # Enhanced thermal boundary layer
        Pe = abs(V_current) * R_current / alpha
        delta_th = np.sqrt(alpha * t) * (1.0 + 0.2 * Pe**0.5)
        
        # Modified heat fluxes
        q_cond = 2 * k * (T_LIQUID_INITIAL - T_prev) / delta_th  # Enhanced conduction
        q_latent = 1.5 * L_VAP * calculate_vapor_density(T_prev) * V_current  # Enhanced latent heat
        q_kinetic = 0.1 * rho * V_current * V_current  # Reduced kinetic contribution
        q_total = q_cond + q_latent + q_kinetic
        
        # Improved temperature change calculation
        c_eff = cp * (1.0 + 0.5 * L_VAP/(rho * cp * T_prev))  # Modified effective heat capacity
        dT = -q_total * dt / (rho * c_eff * delta_th)
        
        # Adaptive temperature change limit
        max_dT = 20.0 * np.sqrt(dt/t)  # Time-dependent limit
        dT = clip_value(dT, -max_dT, max_dT)
        
        # Update with physical bounds
        T_new = T_prev + dT
        return clip_value(T_new, T_INF, T_LIQUID_INITIAL)
        
    except Exception as e:
        return T_prev

@jit(nopython=True)
def integrate_bubble_dynamics(t, R, V, T_s, dt, params):
    """
    RK4 integration with improved stability
    """
    # RK4 integration
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
    
    # Update with stability limits
    R_new = R + (dt/6) * (k1_V + 2*k2_V + 2*k3_V + k4_V)
    V_new = V + (dt/6) * (k1_A + 2*k2_A + 2*k3_A + k4_A)
    
    # Physical bounds matching paper scale with gradual limiting
    R_max = 5e-4  # Increased maximum radius (0.5 mm)
    R_min = 1e-8  # Increased minimum radius
    R_new = max(R_min, min(R_new, R_max))
    
    # Smoother velocity limiting
    V_max = 100.0  # Increased velocity limit
    V_new = clip_value(V_new, -V_max, V_max)
    
    return R_new, V_new

# Modified timestep calculation
@jit(nopython=True)
def calculate_timestep(R, V, T_s, t):
    """
    Adaptive timestep calculation with enhanced stability
    """
    # All relevant timescales
    tau_th = R * R / D_L  # Thermal
    tau_i = R / (abs(V) + 1e-10)  # Inertial
    tau_int = np.sqrt(RHO_L * R**3 / (2 * SIGMA))  # Interface
    
    # Get viscosity for viscous timescale
    mu = get_dynamic_viscosity(T_s)
    tau_v = mu / (delta_p_init + 1e-10)  # Viscous
    
    # Progressive safety factor
    if t < 1e-7:
        safety = 0.001
    elif t < 1e-6:
        safety = 0.01
    else:
        safety = 0.1
        
    dt = safety * min(tau_th, tau_i, tau_int, tau_v)
    return clip_value(dt, 1e-15, 1e-6)

# ---------------------------------------------------------------------
# 1. Physical Constants and Initial Conditions
# ---------------------------------------------------------------------
# Initial conditions for bubble growth
T_SUPERHEAT = 340.1  # K (Case 1 from paper)
T_INF = 1083.6  # K
T_LIQUID_INITIAL = T_INF + T_SUPERHEAT
P_INF = 1.253e5  # Pa

# Physical properties
K_L = 71.7  # W/(m·K)
CP_L = 1380.0  # J/(kg·K)
RHO_L = 927.0  # kg/m³
SIGMA = 0.2  # N/m
D_L = K_L / (RHO_L * CP_L)
L_VAP = 4.1e6  # J/kg

# Critical point properties
T_CRIT = 2503.7  # K
P_CRIT = 25.64e6  # Pa

# Calculate initial conditions with improved early growth model
p_v_init = calculate_vapor_pressure(T_LIQUID_INITIAL)
rho_v_init = calculate_vapor_density(T_LIQUID_INITIAL)

# Enhanced superheat effect for early growth
beta_T = 2.5e-4  # Thermal expansion coefficient
delta_rho = RHO_L * beta_T * T_SUPERHEAT
p_thermal_init = rho_v_init * L_VAP * T_SUPERHEAT / T_LIQUID_INITIAL
p_total_init = p_v_init + p_thermal_init

# Pressure difference drives initial growth
delta_p_init = p_total_init - P_INF

# Initial radius from paper with growth enhancement
R_crit = 2 * SIGMA / delta_p_init
R0 = 2.5e-5  # m (matches paper Case 1)
V0 = 0.01  # Reduced initial velocity

# Time parameters optimized for early growth capture
t_start = 1e-9  # Earlier start time
t_end = 1.0
n_steps = 15000  # More steps for better resolution
max_steps = int(1.2 * n_steps)  # Add buffer for adaptive timestepping

# Calculate initial timestep from physics
tau_th = R0 * R0 / D_L  # Thermal time
tau_i = np.sqrt(RHO_L * R0**3 / (2 * SIGMA))  # Inertial time
tau_v = R0 / np.sqrt(delta_p_init / RHO_L)  # Vapor pressure time
dt_init = 1e-12  # Smaller initial timestep

# Initialize simulation arrays with buffer
t_history = np.zeros(max_steps)
R_history = np.zeros(max_steps)
V_history = np.zeros(max_steps)
Ts_history = np.zeros(max_steps)

# Initialize first elements
t_history[0] = t_start
R_history[0] = R0
V_history[0] = V0
Ts_history[0] = T_LIQUID_INITIAL

# Simulation core
print(f"Initial pressure difference: {delta_p_init/1e5:.2f} bar")
print(f"Critical radius: {R_crit:.3e} m")
print(f"Initial radius: {R0:.3e} m")
print(f"\nStarting simulation:")
print(f"R0={R0:.3e} m")
print(f"T0={T_LIQUID_INITIAL:.1f} K")
print(f"P_inf={P_INF/1e5:.2f} bar")
print(f"Initial p_v={p_v_init/1e5:.3f} bar")

# Main time stepping loop with adaptive dt
R = R0
V = V0
Ts = T_LIQUID_INITIAL
t = t_start
i = 0

for i in tqdm(range(1, n_steps), desc="Simulating"):
    # Calculate adaptive timestep
    dt = calculate_timestep(R, V, Ts, t)
    t = t + dt
    
    # Update temperature first for stability
    Ts = calculate_temperature_plesset_zwick(t, R_history[:i], V_history[:i], t_history[:i], Ts)
    
    # Get temperature-dependent properties
    k = get_thermal_conductivity(Ts)
    rho_v_val = calculate_vapor_density(Ts)
    params = (RHO_L, k, D_L, L_VAP, P_INF, SIGMA, T_INF, rho_v_val)
    
    # Update R and V using RK4
    R_new, V_new = integrate_bubble_dynamics(t, R, V, Ts, dt, params)
    
    # Store history
    t_history[i] = t
    R_history[i] = R_new
    V_history[i] = V_new
    Ts_history[i] = Ts
    
    # Update state
    R = R_new
    V = V_new
    
    # Output progress every 500 steps
    if (i % 500 == 0):
        print(f"\nt={t:.3e} s, R={R:.3e} m, V={V:.2f} m/s, Ts={Ts:.1f} K\n")

print("\nSimulation finished")

# Trim arrays to actual size
t_plot = t_history[:i]
R_plot = R_history[:i] * 100  # Convert to cm
V_plot = V_history[:i] * 100  # Convert to cm/s
T_plot = Ts_history[:i]

# Create figure
fig = plt.figure(figsize=(8, 13))

# Plot 1: Radius vs time (log-log)
ax1 = plt.subplot(311)
ax1.loglog(t_plot, R_plot, 'k-', linewidth=1.5)
ax1.plot(t_plot[::100], R_plot[::100], 'ko', markersize=3, fillstyle='none')
ax1.set_xlim(1e-8, 1)
ax1.set_ylim(1e-4, 1e-1)
ax1.set_ylabel('R (cm)')
ax1.grid(True, which='both', linestyle='-', alpha=0.2)
ax1.grid(True, which='minor', linestyle=':', alpha=0.1)
ax1.set_xticklabels([])

# Plot 2: Temperature vs time (log-x, linear-y)
ax2 = plt.subplot(312)
ax2.semilogx(t_plot, T_plot, 'k-', linewidth=1.5)
ax2.plot(t_plot[::100], T_plot[::100], 'ko', markersize=3, fillstyle='none')
ax2.set_xlim(1e-7, 1)
ax2.set_ylim(1100, 1500)
ax2.set_ylabel('T$_s$ (K)')
ax2.grid(True, which='both', linestyle='-', alpha=0.2)
ax2.grid(True, which='minor', linestyle=':', alpha=0.1)
ax2.set_xticklabels([])

# Plot 3: Velocity vs time (log-log)
ax3 = plt.subplot(313)
ax3.loglog(t_plot, np.abs(V_plot), 'k-', linewidth=1.5)
ax3.plot(t_plot[::100], np.abs(V_plot[::100]), 'ko', markersize=3, fillstyle='none')
ax3.set_xlim(1e-8, 1e-1)
ax3.set_ylim(1e1, 1e4)
ax3.set_xlabel('t (s)')
ax3.set_ylabel('|dR/dt| (cm/s)')
ax3.grid(True, which='both', linestyle='-', alpha=0.2)
ax3.grid(True, which='minor', linestyle=':', alpha=0.1)

# Common formatting
for ax in [ax1, ax2, ax3]:
    ax.tick_params(which='both', direction='in', top=True, right=True)
    
plt.suptitle('Vapor Bubble Growth in Superheated Sodium\n' + 
             f'(T$_∞$ = {T_INF:.1f} K, ΔT = {T_SUPERHEAT:.1f} K)', y=0.95)
plt.tight_layout()
plt.savefig('bubble_dynamics_comparison.png', dpi=300, bbox_inches='tight')