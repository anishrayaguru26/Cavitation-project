import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm

# Temperature-dependent property functions
@jit(nopython=True)
def get_latent_heat(T):
    """
    Get latent heat of vaporization (J/kg) for sodium at a given temperature (K)
    using interpolation from experimental data
    """
    T_data = np.array([
        371, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300,
        1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200,
        2300, 2400, 2500, 2503.7
    ])
    
    L_data = np.array([
        4532, 4510, 4435, 4358, 4279, 4197, 4112, 4025, 3933,
        3838, 3738, 3633, 3523, 3405, 3279, 3143, 2994, 2829,
        2640, 2418, 2141, 1747, 652, 0
    ]) * 1000  # Convert from kJ/kg to J/kg
    
    if T <= T_data[0]:
        return L_data[0]
    if T >= T_data[-1]:
        return L_data[-1]
    
    idx = np.searchsorted(T_data, T)
    T1, T2 = T_data[idx-1], T_data[idx]
    L1, L2 = L_data[idx-1], L_data[idx]
    
    return L1 + (L2 - L1) * (T - T1) / (T2 - T1)

@jit(nopython=True)
def get_liquid_density(T):
    """
    Get liquid density (kg/m³) for sodium at a given temperature (K)
    """
    T_data = np.array([
        400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400,
        1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400,
        2500, 2503.7
    ])
    
    rho_l_data = np.array([
        919, 897, 874, 852, 828, 805, 781, 756, 732, 706, 680,
        653, 626, 597, 568, 537, 504, 469, 431, 387, 335,
        239, 219
    ])
    
    if T <= T_data[0]:
        return rho_l_data[0]
    if T >= T_data[-1]:
        return rho_l_data[-1]
    
    idx = np.searchsorted(T_data, T)
    T1, T2 = T_data[idx-1], T_data[idx]
    rho1, rho2 = rho_l_data[idx-1], rho_l_data[idx]
    
    return rho1 + (rho2 - rho1) * (T - T1) / (T2 - T1)

@jit(nopython=True)
def get_surface_tension(T):
    """
    Calculate surface tension (N/m) for sodium using the formula:
    σ = σ₀(1 - T/Tₒ)ⁿ
    """
    sigma_0 = 0.2405  # N/m
    n = 1.126
    T_c = 2503.7  # Critical temperature in K
    
    T = min(T, T_c)
    sigma = sigma_0 * (1 - T/T_c)**n
    return max(0.0, sigma)

@jit(nopython=True)
def get_vapor_pressure(T):
    """
    Calculate vapor pressure (Pa) for sodium using the formula:
    ln P = 11.9463 - 12633.73/T - 0.4672 ln T
    """
    ln_P = 11.9463 - 12633.73/T - 0.4672 * np.log(T)
    P = np.exp(ln_P) * 1e6  # Convert MPa to Pa
    return P

@jit(nopython=True)
def get_vapor_density(T):
    """
    Get vapor density (kg/m³) for sodium at a given temperature (K)
    """
    T_data = np.array([
        400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400,
        1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400,
        2500, 2503.7
    ])
    
    rho_v_data = np.array([
        1.24e-9, 5.03e-7, 2.63e-5, 4.31e-4, 3.43e-3, 1.70e-2,
        6.03e-2, 0.168, 0.394, 0.805, 1.48, 2.50, 3.96, 5.95,
        8.54, 11.9, 16.0, 21.2, 27.7, 36.3, 49.3, 102.0, 219.0
    ])
    
    if T <= T_data[0]:
        return rho_v_data[0]
    if T >= T_data[-1]:
        return rho_v_data[-1]
    
    idx = np.searchsorted(T_data, T)
    T1, T2 = T_data[idx-1], T_data[idx]
    rho1, rho2 = rho_v_data[idx-1], rho_v_data[idx]
    
    log_rho = np.log(rho1) + (np.log(rho2) - np.log(rho1)) * (T - T1) / (T2 - T1)
    return np.exp(log_rho)

# Add helper functions with JIT compilation
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
def rk4_step(R, V, T_s, dt, params):
    """
    4th order Runge-Kutta integration with proper coupling
    """
    # k1 calculations
    k1_R = V
    _, k1_V = rayleigh_plesset_optimized(R, V, 0.0, T_s, params)
    
    # k2 calculations
    R2 = R + 0.5 * dt * k1_R
    V2 = V + 0.5 * dt * k1_V
    _, k2_V = rayleigh_plesset_optimized(R2, V2, dt/2, T_s, params)
    k2_R = V2
    
    # k3 calculations
    R3 = R + 0.5 * dt * k2_R
    V3 = V + 0.5 * dt * k2_V
    _, k3_V = rayleigh_plesset_optimized(R3, V3, dt/2, T_s, params)
    k3_R = V3
    
    # k4 calculations
    R4 = R + dt * k3_R
    V4 = V + dt * k3_V
    _, k4_V = rayleigh_plesset_optimized(R4, V4, dt, T_s, params)
    k4_R = V4
    
    # Final update with proper weighting
    R_new = R + (dt/6.0) * (k1_R + 2*k2_R + 2*k3_R + k4_R)
    V_new = V + (dt/6.0) * (k1_V + 2*k2_V + 2*k3_V + k4_V)
    
    return R_new, V_new

@jit(nopython=True)
def rayleigh_plesset_optimized(R, dR_dt, t, T_s, params):
    """
    Exact Rayleigh-Plesset equation with inertial effects
    """
    rho_l, k, D, L, p_inf, sigma, T_inf, rho_v = params
    
    # Basic stability checks
    R = max(R, 1e-12)
    T_s = clip_value(T_s, T_MIN, T_MAX)
    
    # Vapor pressure with superheat effect
    p_v = get_vapor_pressure(T_s)
    
    # Thermal pressure term (from kinetic theory)
    p_thermal = rho_v * L * (T_s - T_inf) / T_s
    
    # Surface tension with dynamic effects
    p_surface = 2 * sigma / R - sigma * dR_dt / (R * R)
    
    # Viscous effects with temperature dependence
    mu = 2.29e-4 * (T_s/1154.0)**0.5
    p_viscous = 4 * mu * dR_dt / R
    
    # Total pressure difference
    delta_p = p_v + p_thermal - p_inf - p_surface - p_viscous
    
    # Full Rayleigh-Plesset equation with all terms
    R_ddot = (delta_p / (rho_l * R)) - (3 * dR_dt * dR_dt) / (2 * R) - (4 * mu * dR_dt) / (rho_l * R * R)
    
    return dR_dt, R_ddot

def calculate_temperature_exact(t, R_hist, dRdt_hist, t_hist, T_prev):
    """
    Calculate temperature with Plesset-Zwick solution
    """
    if len(t_hist) < 3:
        return T_LIQUID_INITIAL
        
    try:
        # Get current state
        R_current = R_hist[-1]
        dRdt_current = dRdt_hist[-1]
        dt = t - t_hist[-2]
        
        # Calculate interfacial temperature change
        alpha = K_L / (RHO_L * CP_L)  # Thermal diffusivity
        thermal_bl = np.sqrt(alpha * t)  # Thermal boundary layer thickness
        
        # Heat flux through bubble wall
        q = K_L * (T_LIQUID_INITIAL - T_prev) / thermal_bl
        
        # Mass flux from phase change
        j = q / L_VAP
        
        # Temperature change from energy balance
        dT = -j * L_VAP * dt / (RHO_L * CP_L * thermal_bl)
        
        # Update temperature with physical bounds
        T_new = T_prev + dT
        return clip_value(T_new, T_MIN, T_LIQUID_INITIAL)
        
    except Exception as e:
        return T_prev

# Add minimum and maximum temperature bounds as constants
T_MIN = 300.0  # K (minimum reasonable temperature)
T_MAX = 2500.0  # K (close to critical point of sodium)

# ---------------------------------------------------------------------
# 1. Physical Properties with improved temperature dependence
# ---------------------------------------------------------------------
P_INF = 0.5e5  # Pa (0.5 bar)
T_SUPERHEAT = 340.0 # K
T_INF = 881.0 + 273 # K (Bulk temperature)
T_LIQUID_INITIAL = T_INF + T_SUPERHEAT

# Improved physical properties
K_L = 71.7     # W/(m*K) (Liquid thermal conductivity at mean temperature)
CP_L = 1300.0  # J/(kg*K) (Liquid specific heat - more accurate value)
RHO_L = get_liquid_density(T_LIQUID_INITIAL)
SIGMA = get_surface_tension(T_LIQUID_INITIAL)
D_L = K_L / (RHO_L * CP_L)  # Thermal diffusivity
L_VAP = get_latent_heat(T_LIQUID_INITIAL)

# --- Vapor Properties (Highly Temperature Dependent!) ---
def p_v(T):
    return get_vapor_pressure(T)

def rho_v(T):
    return get_vapor_density(T)

# ---------------------------------------------------------------------
# 2. Initial Conditions
# ---------------------------------------------------------------------
# Update initial conditions for better stability
R0 = 2e-6  # Initial radius (2 micrometers)
V0 = 1e-3  # Small initial velocity to break symmetry
Ts0 = T_LIQUID_INITIAL  # Start at initial liquid temperature

print(f"Initial radius: {R0:.3e} m")
print(f"Initial velocity: {V0:.3e} m/s")
print(f"Initial temperature: {Ts0:.1f} K")

# ---------------------------------------------------------------------
# 3. Simulation Parameters
# ---------------------------------------------------------------------
t_start = 1e-10  # Start time (s)
t_end = 1.0      # End time (s)
n_steps = 10000  # Increase number of steps for better accuracy
dt = (t_end - t_start) / n_steps

# Pre-allocate larger arrays for history storage
t_history = np.zeros(n_steps + 1)
R_history = np.zeros(n_steps + 1)
V_history = np.zeros(n_steps + 1)
Ts_history = np.zeros(n_steps + 1)

# Initialize first elements with exact values
t_history[0] = t_start
R_history[0] = R0
V_history[0] = V0
Ts_history[0] = Ts0

# Main time stepping loop
R = R0
V = V0
Ts = Ts0
t = t_start

print(f"Starting simulation: R0={R0:.3e} m, Ts0={Ts0:.1f} K, P_inf={P_INF/1e5:.2f} bar")
print(f"Initial p_v(Ts0)={p_v(Ts0)/1e5:.3f} bar")

for i in tqdm(range(1, n_steps + 1), desc="Simulating bubble dynamics", unit="step"):
    if R <= 0:
        print(f"\nBubble collapsed at t={t:.3e} s")
        break
        
    t = t_history[i] = t_start + i * dt
    
    # Update temperature first
    Ts = calculate_temperature_exact(t, R_history[:i], V_history[:i], t_history[:i], Ts)
    
    # Update physical properties
    rho_l = get_liquid_density(Ts)
    rho_v_val = get_vapor_density(Ts)
    L = get_latent_heat(Ts)
    sigma = get_surface_tension(Ts)
    
    params = (rho_l, K_L, D_L, L, P_INF, sigma, T_INF, rho_v_val)
    
    # RK4 integration for radius and velocity
    R_new, V_new = rk4_step(R, V, Ts, dt, params)
    
    # Update state
    R = max(R_new, 1e-12)  # Ensure positive radius
    V = clip_value(V_new, -1e6, 1e6)  # Limit velocity to physical values
    
    # Store history
    R_history[i] = R
    V_history[i] = V
    Ts_history[i] = Ts
    
    if (i + 1) % (n_steps // 20) == 0:
        tqdm.write(f"t={t:.3e} s, R={R:.3e} m, V={V:.2f} m/s, Ts={Ts:.1f} K")

print("\nSimulation finished.")

# ---------------------------------------------------------------------
# 6. Plotting Results
# ---------------------------------------------------------------------
t_history = np.array(t_history)
R_history = np.array(R_history)
V_history = np.array(V_history)
Ts_history = np.array(Ts_history)

fig, axs = plt.subplots(3, 1, figsize=(12, 15))

# Plot 1: Radius (log-log scale with raw data)
axs[0].loglog(t_history, R_history, 'b-', linewidth=1, label='Radius')
axs[0].loglog(t_history[::10], R_history[::10], 'k.', markersize=2, alpha=0.5)
axs[0].set_ylabel("Radius [m]")
axs[0].grid(True, which='both', ls='-', alpha=0.2)
axs[0].legend()

# Plot 2: Velocity (log-log scale with raw data)
vel_abs = np.abs(V_history)
axs[1].loglog(t_history, vel_abs, 'r-', linewidth=1, label='|Velocity|')
axs[1].loglog(t_history[::10], vel_abs[::10], 'k.', markersize=2, alpha=0.5)
axs[1].set_ylabel("Velocity [m/s]")
axs[1].grid(True, which='both', ls='-', alpha=0.2)
axs[1].legend()

# Plot 3: Temperature (log-linear scale with raw data)
axs[2].semilogx(t_history, Ts_history, 'g-', linewidth=1, label='Temperature')
axs[2].plot(t_history[::10], Ts_history[::10], 'k.', markersize=2, alpha=0.5)
axs[2].set_ylabel("Temperature [K]")
axs[2].set_xlabel("Time [s]")
axs[2].grid(True, which='both', ls='-', alpha=0.2)
axs[2].axhline(T_LIQUID_INITIAL, color='r', ls='--', label=f'T_initial ({T_LIQUID_INITIAL:.1f} K)')
axs[2].axhline(T_INF, color='b', ls='--', label=f'T_bulk ({T_INF:.1f} K)')
axs[2].legend()

plt.suptitle(f"Bubble Dynamics (Na, {P_INF/1e5:.1f} bar, {T_SUPERHEAT:.0f} K Superheat)")
plt.tight_layout()
plt.show()