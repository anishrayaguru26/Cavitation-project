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
def rayleigh_plesset_optimized(R, dR_dt, t, T_s, params):
    """
    Optimized Rayleigh-Plesset equation calculation
    """
    rho_l, k, D, L, p_inf, sigma, T_inf, rho_v = params
    
    # Prevent division by zero
    R = max(R, 1e-12)
    
    # Get vapor pressure
    ln_P = 11.9463 - 12633.73/T_s - 0.4672 * np.log(T_s)
    p_v = np.exp(ln_P) * 1e6
    
    # Calculate dynamic viscosity
    mu = 2.8e-4
    
    # Surface tension pressure
    p_surface = 2 * sigma / R
    
    # Viscous term
    p_viscous = 4 * mu * dR_dt / R
    
    # Pressure difference
    delta_p = p_v - p_inf - p_surface - p_viscous
    
    # Acceleration terms
    R_ddot = (delta_p / (rho_l * R)) - (3 * dR_dt * dR_dt) / (2 * R)
    
    return dR_dt, R_ddot

# Add minimum and maximum temperature bounds as constants
T_MIN = 300.0  # K (minimum reasonable temperature)
T_MAX = 2500.0  # K (close to critical point of sodium)

def calculate_Ts_integral_term(t_hist, R_hist, Ts_hist, current_t):
    """
    Vectorized calculation of the integral term in the Ts equation with improved stability
    """
    if len(t_hist) < 3:
        return T_INF

    try:
        # Convert to numpy arrays once and ensure minimum positive values
        t_arr = np.asarray(t_hist, dtype=np.float64)
        R_arr = np.maximum(np.asarray(R_hist, dtype=np.float64), 1e-12)
        Ts_arr = np.clip(np.asarray(Ts_hist, dtype=np.float64), T_MIN, T_MAX)
        
        # Vectorized calculation of ρv with stability checks
        rho_v_hist = np.maximum(np.vectorize(rho_v)(Ts_arr), 1e-12)
        
        # Calculate R³ρv with numerical stability
        R_cubed = np.clip(R_arr**3, 1e-36, 1e6)
        integrand_part1_base = R_cubed * rho_v_hist
        
        # Calculate derivative with bounds
        try:
            deriv_term = np.gradient(integrand_part1_base, t_arr, edge_order=1)
            deriv_term = np.clip(deriv_term, -1e10, 1e10)  # Limit extreme derivatives
        except:
            return T_INF
        
        # Calculate inner integral with improved stability
        R4_values = np.clip(R_arr**4, 1e-48, 1e8)
        mask = (t_arr <= current_t)
        
        if not np.any(mask):
            return T_INF
            
        integral = np.trapz(R4_values[mask], t_arr[mask])
        inner_integral = max(integral, 1e-30)
            
        # Calculate final temperature with strict bounds
        delta_T = (1.0 / (3.0 * K_L)) * np.sqrt(D_L / np.pi) * safe_divide(deriv_term[-1], np.sqrt(inner_integral), 0.0)
        delta_T = np.clip(delta_T, -200, 200)  # Limit maximum temperature change to physically reasonable values
        
        T_s = T_INF - delta_T
        return np.clip(T_s, T_MIN, T_MAX)
            
    except Exception as e:
        return T_INF

@jit(nopython=True)
def integrate_substeps(R_current, dR_dt_current, dt_sub, n_substeps, params, T_s):
    """Optimized sub-step integration with improved stability"""
    for _ in range(n_substeps):
        # Ensure positive radius
        R_current = max(R_current, 1e-12)
        
        # Calculate new derivatives
        dR_dt_new, R_ddot = rayleigh_plesset_optimized(R_current, dR_dt_current, 0.0, T_s, params)
        
        # Handle invalid values
        if not (np.isfinite(dR_dt_new) and np.isfinite(R_ddot)):
            dR_dt_new = dR_dt_current
            R_ddot = 0.0
        
        # Update with strict bounds
        dR_dt_new = clip_value(dR_dt_new, -1e6, 1e6)
        R_new = max(1e-12, min(1e-3, R_current + dt_sub * dR_dt_new))  # Limit radius between 1pm and 1mm
        dR_dt_current = clip_value(dR_dt_new + dt_sub * R_ddot, -1e6, 1e6)
        R_current = R_new
        
    return R_current, dR_dt_current

# ---------------------------------------------------------------------
# 1. Define Physical Properties for Liquid Sodium (PLACEHOLDER VALUES)
#    These MUST be replaced with accurate, potentially temperature-dependent
#    values for the specified conditions (p=0.5 bar, T_superheat=340K).
# ---------------------------------------------------------------------
P_INF = 0.5e5  # Pa (0.5 bar)
T_SUPERHEAT = 340.0 # K
# Need saturation temperature at P_INF to find T_liquid
# T_sat_Na_at_05bar = ... # ~900-1000 K range? Needs lookup.
# T_INF = T_sat_Na_at_05bar # Bulk liquid temperature (assume saturated)
T_INF = 881.0 + 273 # K (Assumed bulk temperature - needs accurate value)
T_LIQUID_INITIAL = T_INF + T_SUPERHEAT # Initial superheated temp

RHO_L = get_liquid_density(T_LIQUID_INITIAL)
SIGMA = get_surface_tension(T_LIQUID_INITIAL)
K_L = 60.0     # W/(m*K) (Liquid thermal conductivity - temp dependent!)
CP_L = 1250.0  # J/(kg*K) (Liquid specific heat - temp dependent!)
D_L = K_L / (RHO_L * CP_L) # Liquid thermal diffusivity
L_VAP = get_latent_heat(T_LIQUID_INITIAL)

# --- Vapor Properties (Highly Temperature Dependent!) ---
def p_v(T):
    return get_vapor_pressure(T)

def rho_v(T):
    return get_vapor_density(T)

# ---------------------------------------------------------------------
# 2. Initial Conditions
# ---------------------------------------------------------------------
R0 = 1e-6  # Initial radius of 1 micrometer (m)
V0 = 0.0  # m/s (Initially at rest)
Ts0 = T_LIQUID_INITIAL # K

# ---------------------------------------------------------------------
# 3. Simulation Parameters
# ---------------------------------------------------------------------
t_start = 1e-10
t_end = 1.0  # s
dt = (t_end - t_start) / 1000  # Reduce steps from 10000 to 1000 
n_steps = int((t_end - t_start) / dt)

def calculate_dVdt(R, V, Ts):
    """Calculates dV/dt = d²R/dt² from Rayleigh-Plesset with improved stability."""
    if R < 1e-12:  # Avoid division by zero
        return 0.0
    try:
        pv = p_v(Ts)
        term1 = safe_divide(pv - P_INF - 2 * SIGMA / max(R, 1e-12), RHO_L)
        term2 = 1.5 * (V**2)
        dVdt = safe_divide(term1 - term2, max(R, 1e-12))
        
        # Limit acceleration to prevent numerical instability
        return np.clip(dVdt, -1e8, 1e8)
    except:
        return 0.0

# Pre-allocate arrays for history storage at start
t_history = np.zeros(n_steps + 1)
R_history = np.zeros(n_steps + 1)
V_history = np.zeros(n_steps + 1)
Ts_history = np.zeros(n_steps + 1)

# Initialize first elements
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
    
    # Calculate derivatives at current state
    dVdt = calculate_dVdt(R, V, Ts)
    
    # Update R and V with Forward Euler
    R_new = R + V * dt
    V_new = V + dVdt * dt
    
    # Calculate surface temperature
    integral_term_val = calculate_Ts_integral_term(t_history[:i], R_history[:i], Ts_history[:i], t)
    
    if np.isnan(integral_term_val) or np.isinf(integral_term_val):
        tqdm.write(f"Warning: Invalid integral term ({integral_term_val:.3e}) at t={t:.3e} s. Holding Ts constant.")
        Ts_new = Ts
    else:
        Ts_new = T_INF - (1.0 / (3.0 * K_L)) * np.sqrt(D_L / np.pi) * integral_term_val
    
    # Update temperature-dependent properties
    rho_l = get_liquid_density(Ts_new)
    rho_v_val = get_vapor_density(Ts_new)
    L = get_latent_heat(Ts_new)
    sigma = get_surface_tension(Ts_new)
    
    params = (rho_l, K_L, D_L, L, P_INF, sigma, T_INF, rho_v_val)
    
    # Adaptive sub-stepping
    vel_factor = abs(V_new) * dt / (0.01 * max(R_new, 1e-12))
    acc_factor = abs(dVdt) * dt * dt / (0.01 * max(R_new, 1e-12))
    n_substeps = max(1, min(1000, int(np.ceil(max(vel_factor, acc_factor)))))
    dt_sub = dt / n_substeps
    
    # Perform sub-step integration
    R_new, V_new = integrate_substeps(R_new, V_new, dt_sub, n_substeps, params, Ts_new)
    
    # Update state
    R = R_new
    V = V_new
    Ts = Ts_new
    
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

fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Plot 1: Radius (log-log scale)
axs[0].loglog(t_history, R_history)
axs[0].set_ylabel("Radius [m]")
axs[0].grid(True, which='both', linestyle='-', alpha=0.6)
axs[0].grid(True, which='minor', linestyle=':', alpha=0.3)
axs[0].set_title('Bubble Radius Evolution')

# Plot 2: Velocity (log-log scale with absolute value)
axs[1].loglog(t_history, np.abs(V_history) + 1e-10)  # Add small value to handle zero velocity
axs[1].set_ylabel("Velocity [m/s]")
axs[1].grid(True, which='both', linestyle='-', alpha=0.6)
axs[1].grid(True, which='minor', linestyle=':', alpha=0.3)
axs[1].set_title('Bubble Wall Velocity')

# Plot 3: Temperature (linear scale with log time)
axs[2].semilogx(t_history, Ts_history, 'r-', linewidth=2)
axs[2].set_ylabel("Surface Temperature [K]")
axs[2].set_xlabel("Time [s]")
axs[2].grid(True, which='both', linestyle='-', alpha=0.6)
axs[2].grid(True, which='minor', linestyle=':', alpha=0.3)
axs[2].set_title('Surface Temperature Evolution')

# Add reference lines for temperature plot
axs[2].axhline(T_LIQUID_INITIAL, color='r', linestyle='--', label=f'T_initial ({T_LIQUID_INITIAL:.1f} K)')
axs[2].axhline(T_INF, color='g', linestyle='--', label=f'T_bulk ({T_INF:.1f} K)')
axs[2].legend()

# Adjust layout and add overall title
plt.suptitle(f"Bubble Dynamics (Na, {P_INF/1e5:.1f} bar, {T_SUPERHEAT:.0f} K Superheat)")
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()