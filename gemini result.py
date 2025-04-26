import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from numba import jit

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
dt = (t_end - t_start) / 10000  # Adjust number of steps for smooth results
n_steps = int((t_end - t_start) / dt)

# History storage with pre-allocated arrays for better performance
t_history = [t_start]
R_history = [R0]
V_history = [V0]
Ts_history = [Ts0]

# Add helper function for numerical stability
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

def calculate_Ts_integral_term(t_hist, R_hist, Ts_hist, current_t):
    """
    Numerically estimates the integral term in the Ts equation with improved stability
    """
    if len(t_hist) < 3:
        return 0.0

    try:
        # Convert lists to numpy arrays with error checking
        t_arr = np.array(t_hist, dtype=np.float64)
        R_arr = np.array(R_hist, dtype=np.float64)
        Ts_arr = np.array(Ts_hist, dtype=np.float64)
        
        # Ensure minimum positive values for stability
        R_arr = np.maximum(R_arr, 1e-12)
        
        # Calculate ρv with error handling
        rho_v_hist = np.array([max(rho_v(T), 1e-12) for T in Ts_arr])
        
        # Calculate R³ρv with stability checks
        R_cubed = R_arr**3
        integrand_part1_base = R_cubed * rho_v_hist
        
        # Calculate derivative with error handling
        if len(t_arr) > 1:
            try:
                deriv_term = np.gradient(integrand_part1_base, t_arr, edge_order=1)
            except:
                # Fallback to forward differences if gradient fails
                deriv_term = np.zeros_like(t_arr)
                dt = np.diff(t_arr)
                deriv_term[1:] = np.diff(integrand_part1_base) / dt
                deriv_term[0] = deriv_term[1]
        else:
            return 0.0
            
        # Improved interpolation with bounds checking
        if len(t_arr) > 1:
            try:
                interp_R = interpolate.interp1d(t_arr, R_arr, kind='linear', 
                                              bounds_error=False, fill_value=(R_arr[0], R_arr[-1]))
            except:
                def interp_R(t_val):
                    return R_arr[0]
        else:
            def interp_R(t_val):
                return R_arr[0]
                
        # Inner integral calculation with improved stability
        def inner_integral_sqrt_inv(x_val, t_now):
            if x_val >= t_now - 1e-15:
                return 0.0
            try:
                integral_R4, err = integrate.quad(
                    lambda y: max(interp_R(y), 1e-12)**4,
                    x_val, t_now,
                    limit=50,
                    epsabs=1e-6,
                    epsrel=1e-6
                )
                if integral_R4 <= 1e-30:
                    return 0.0
                return 1.0 / np.sqrt(max(integral_R4, 1e-30))
            except:
                return 0.0
                
        # Outer integral with improved stability
        try:
            integral_val, err = integrate.quad(
                lambda x: L_VAP * safe_divide(
                    deriv_term[np.searchsorted(t_arr, x) - 1] * inner_integral_sqrt_inv(x, current_t),
                    1.0
                ),
                t_arr[0],
                current_t,
                limit=50,
                epsabs=1e-6,
                epsrel=1e-6
            )
            
            # Ensure result is finite and reasonable
            if not np.isfinite(integral_val):
                return 0.0
            return np.clip(integral_val, -1e10, 1e10)
            
        except Exception as e:
            return 0.0
            
    except Exception as e:
        return 0.0

# ---------------------------------------------------------------------
# 5. Time Stepping Loop
# ---------------------------------------------------------------------
R = R0
V = V0
Ts = Ts0
t = t_start

print(f"Starting simulation: R0={R0:.3e} m, Ts0={Ts0:.1f} K, P_inf={P_INF/1e5:.2f} bar")
print(f"Initial p_v(Ts0)={p_v(Ts0)/1e5:.3f} bar")

for i in range(n_steps):
    if R <= 0:
        print(f"Bubble collapsed at t={t:.3e} s")
        break

    # Calculate derivatives at current state
    dVdt = calculate_dVdt(R, V, Ts)
    # dRdt = V # Already have V

    # --- Update R and V (Forward Euler - consider RK4 for better stability) ---
    R_new = R + V * dt
    V_new = V + dVdt * dt

    # --- Update Ts ---
    # This is the expensive part - requires history
    integral_term_val = calculate_Ts_integral_term(t_history, R_history, Ts_history, t)

    # Check for issues in integral calculation
    if np.isnan(integral_term_val) or np.isinf(integral_term_val):
         print(f"Warning: Invalid integral term ({integral_term_val:.3e}) at t={t:.3e} s. Holding Ts constant.")
         Ts_new = Ts # Keep previous value if calculation fails
    else:
         Ts_new = T_INF - (1.0 / (3.0 * K_L)) * np.sqrt(D_L / np.pi) * integral_term_val

    # Update time and state
    t += dt
    R = R_new
    V = V_new
    Ts = Ts_new

    # Store history
    t_history.append(t)
    R_history.append(R)
    V_history.append(V)
    Ts_history.append(Ts)

    if (i + 1) % (n_steps // 20) == 0: # Print progress
        print(f"t={t:.3e} s, R={R:.3e} m, V={V:.2f} m/s, Ts={Ts:.1f} K, IntegralTerm={integral_term_val:.3e}")


print("Simulation finished.")

# ---------------------------------------------------------------------
# 6. Plotting Results
# ---------------------------------------------------------------------
t_history = np.array(t_history)
R_history = np.array(R_history)
V_history = np.array(V_history)
Ts_history = np.array(Ts_history)

fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

axs[0].plot(t_history * 1000, R_history * 1000)
axs[0].set_ylabel("Radius R (mm)")
axs[0].grid(True)

axs[1].plot(t_history * 1000, V_history)
axs[1].set_ylabel("Velocity V (m/s)")
axs[1].grid(True)

axs[2].plot(t_history * 1000, Ts_history)
axs[2].set_ylabel("Surface Temp Ts (K)")
axs[2].set_xlabel("Time (ms)")
axs[2].grid(True)
# Add reference lines if useful (e.g., T_INF, T_LIQUID_INITIAL)
axs[2].axhline(T_LIQUID_INITIAL, color='r', linestyle='--', label=f'T_initial ({T_LIQUID_INITIAL:.1f} K)')
axs[2].axhline(T_INF, color='g', linestyle='--', label=f'T_bulk ({T_INF:.1f} K)')
axs[2].legend()


plt.suptitle(f"Bubble Dynamics (Na, {P_INF/1e5:.1f} bar, {T_SUPERHEAT:.0f} K Superheat)")
plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
plt.show()