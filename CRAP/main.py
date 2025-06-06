import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
from numba import jit, float64, prange
import os
import sys

@jit(nopython=True)
def isfinite(x):
    """Numba-compatible implementation of np.isfinite"""
    return not (np.isinf(x) or np.isnan(x))

@jit(nopython=True)
def calculate_gradient(y, x):
    """
    Calculate gradient using central differences with Numba optimization and enhanced stability
    """
    n = len(y)
    dy = np.zeros(n)
    
    try:
        # Forward difference for first point with stability check
        dx_start = max(x[1] - x[0], 1e-15)  # Prevent division by zero
        dy[0] = (y[1] - y[0]) / dx_start
        if not isfinite(dy[0]):
            dy[0] = 0.0
        
        # Central difference for interior points with stability checks
        for i in range(1, n-1):
            dx = max(x[i+1] - x[i-1], 1e-15)  # Prevent division by zero
            dy[i] = (y[i+1] - y[i-1]) / dx
            if not isfinite(dy[i]):
                dy[i] = dy[i-1] if isfinite(dy[i-1]) else 0.0
        
        # Backward difference for last point with stability check
        dx_end = max(x[-1] - x[-2], 1e-15)  # Prevent division by zero
        dy[-1] = (y[-1] - y[-2]) / dx_end
        if not isfinite(dy[-1]):
            dy[-1] = dy[-2] if isfinite(dy[-2]) else 0.0
            
    except:
        # If any calculation fails, return zeros
        return np.zeros(n)
    
    # Final check for any remaining invalid values
    for i in range(n):
        if not isfinite(dy[i]):
            dy[i] = 0.0
            
    return dy

@jit(nopython=True)
def calculate_inner_integral(times, radii, start_idx):
    """Optimized calculation of inner integral with enhanced stability"""
    n = len(times)
    if start_idx >= n - 1:
        return 1e-30
        
    try:
        integral = 0.0
        prev_valid_value = 0.0
        
        for i in range(start_idx + 1, n):
            dt = max(times[i] - times[i-1], 1e-15)  # Prevent zero time step
            
            # Safely calculate R^4 averages with bounds
            r1 = max(min(radii[i], 1e-3), 1e-12)  # Limit radius between 1pm and 1mm
            r2 = max(min(radii[i-1], 1e-3), 1e-12)
            
            R4_1 = r1**4
            R4_2 = r2**4
            
            if isfinite(R4_1) and isfinite(R4_2):
                R4_avg = (R4_1 + R4_2) / 2
                increment = R4_avg * dt
                if isfinite(increment):
                    integral += increment
                    prev_valid_value = integral
            else:
                integral = prev_valid_value
                
    except:
        return max(prev_valid_value, 1e-30)
        
    return max(integral, 1e-30)  # Ensure positive value

@jit(nopython=True)
def get_latent_heat(T):
    """
    Get latent heat of vaporization (J/kg) for sodium at a given temperature (K)
    using interpolation from experimental data
    """
    # Temperature points (K)
    T_data = np.array([
        371, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300,
        1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200,
        2300, 2400, 2500, 2503.7
    ])
    
    # Latent heat points (J/kg)
    L_data = np.array([
        4532, 4510, 4435, 4358, 4279, 4197, 4112, 4025, 3933,
        3838, 3738, 3633, 3523, 3405, 3279, 3143, 2994, 2829,
        2640, 2418, 2141, 1747, 652, 0
    ]) * 1000  # Convert from kJ/kg to J/kg
    
    # Check if temperature is within bounds
    if T <= T_data[0]:
        return L_data[0]
    if T >= T_data[-1]:
        return L_data[-1]
    
    # Find indices for interpolation
    idx = np.searchsorted(T_data, T)
    
    # Linear interpolation
    T1, T2 = T_data[idx-1], T_data[idx]
    L1, L2 = L_data[idx-1], L_data[idx]
    
    return L1 + (L2 - L1) * (T - T1) / (T2 - T1)

@jit(nopython=True)
def get_liquid_density(T):
    """
    Get liquid density (kg/m³) for sodium at a given temperature (K)
    using interpolation from experimental data
    """
    # Temperature points (K)
    T_data = np.array([
        400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400,
        1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400,
        2500, 2503.7
    ])
    
    # Liquid density points (kg/m³)
    rho_l_data = np.array([
        919, 897, 874, 852, 828, 805, 781, 756, 732, 706, 680,
        653, 626, 597, 568, 537, 504, 469, 431, 387, 335,
        239, 219
    ])
    
    # Check if temperature is within bounds
    if T <= T_data[0]:
        return rho_l_data[0]
    if T >= T_data[-1]:
        return rho_l_data[-1]
    
    # Find indices for interpolation
    idx = np.searchsorted(T_data, T)
    
    # Linear interpolation
    T1, T2 = T_data[idx-1], T_data[idx]
    rho1, rho2 = rho_l_data[idx-1], rho_l_data[idx]
    
    return rho1 + (rho2 - rho1) * (T - T1) / (T2 - T1)

@jit(nopython=True)
def get_vapor_density(T):
    """
    Get vapor density (kg/m³) for sodium at a given temperature (K)
    using interpolation from experimental data
    """
    # Temperature points (K)
    T_data = np.array([
        400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400,
        1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400,
        2500, 2503.7
    ])
    
    # Vapor density points (kg/m³)
    rho_v_data = np.array([
        1.24e-9, 5.03e-7, 2.63e-5, 4.31e-4, 3.43e-3, 1.70e-2,
        6.03e-2, 0.168, 0.394, 0.805, 1.48, 2.50, 3.96, 5.95,
        8.54, 11.9, 16.0, 21.2, 27.7, 36.3, 49.3, 102.0, 219.0
    ])
    
    # Check if temperature is within bounds
    if T <= T_data[0]:
        return rho_v_data[0]
    if T >= T_data[-1]:
        return rho_v_data[-1]
    
    # Find indices for interpolation
    idx = np.searchsorted(T_data, T)
    
    # Linear interpolation
    T1, T2 = T_data[idx-1], T_data[idx]
    rho1, rho2 = rho_v_data[idx-1], rho_v_data[idx]
    
    # Use log interpolation for vapor density since it varies over many orders of magnitude
    log_rho = np.log(rho1) + (np.log(rho2) - np.log(rho1)) * (T - T1) / (T2 - T1)
    return np.exp(log_rho)

@jit(nopython=True)
def get_surface_tension(T):
    """
    Calculate surface tension (N/m) for sodium using the formula:
    σ = σ₀(1 - T/Tₒ)ⁿ
    where:
    σ₀ = 240.5 mN/m = 0.2405 N/m
    n = 1.126
    Tₒ = 2503.7 K
    """
    sigma_0 = 0.2405  # Convert from mN/m to N/m
    n = 1.126
    T_c = 2503.7  # Critical temperature in K
    
    # Ensure temperature doesn't exceed critical point
    T = min(T, T_c)
    
    # Calculate surface tension
    sigma = sigma_0 * (1 - T/T_c)**n
    
    # Ensure non-negative surface tension
    return max(0.0, sigma)

@jit(nopython=True)
def get_vapor_pressure(T):
    """
    Calculate vapor pressure (Pa) for sodium using the formula:
    ln P = 11.9463 - 12633.73/T - 0.4672 ln T
    where P is in MPa and T is in K
    """
    # Calculate ln(P) where P is in MPa
    ln_P = 11.9463 - 12633.73/T - 0.4672 * np.log(T)
    
    # Convert from ln(MPa) to Pa
    P = np.exp(ln_P) * 1e6  # Convert MPa to Pa
    
    return P

@jit(nopython=True)
def calculate_Ts_optimized(t, times, radii, T_inf, k, D, rho_v, superheat):
    """
    Calculate surface temperature T_s using the integral equation with improved stability
    """
    if len(times) < 2:
        return T_inf
    
    # Ensure minimum radius values
    n = len(radii)
    radii_safe = np.zeros(n)
    for i in range(n):
        radii_safe[i] = max(radii[i], 1e-12)
    
    # Calculate R³ρᵥ term with stability check
    R_cubed_rho_v = np.zeros(n)
    for i in range(n):
        R_cubed_rho_v[i] = radii_safe[i]**3 * rho_v
        
    # Calculate derivative with stability check
    if len(times) > 1:
        try:
            deriv_term = calculate_gradient(R_cubed_rho_v, times)
        except:
            return T_inf
    else:
        return T_inf
    
    # Calculate inner integral
    integral_term = calculate_inner_integral(times, radii_safe, 0)
    if integral_term <= 1e-30:
        return T_inf
    
    # Calculate temperature change
    delta_T = (1.0 / (3.0 * k)) * np.sqrt(D / np.pi) * deriv_term[-1] / np.sqrt(integral_term)
    
    # Apply temperature change with bounds
    T_s = T_inf - delta_T
    
    # Apply temperature bounds
    if T_s < 0.5 * T_inf:
        T_s = 0.5 * T_inf
    elif T_s > 1.5 * T_inf:
        T_s = 1.5 * T_inf
    
    return float(T_s)

@jit(nopython=True)
def rayleigh_plesset_optimized(R, dR_dt, t, T_s, params):
    """
    Optimized Rayleigh-Plesset equation calculation using interpolated vapor pressure
    """
    rho_l, k, D, L, p_inf, sigma, T_inf, rho_v = params
    
    # Prevent division by zero and ensure numerical stability
    R = max(R, 1e-12)
    
    # Get vapor pressure from temperature using formula
    p_v = get_vapor_pressure(T_s)
    
    # Calculate dynamic viscosity (Pa⋅s)
    mu = 2.8e-4  # Dynamic viscosity of liquid sodium
    
    # Surface tension pressure
    p_surface = 2 * sigma / R
    
    # Viscous term
    p_viscous = 4 * mu * dR_dt / R
    
    # Pressure difference driving bubble growth
    delta_p = p_v - p_inf - p_surface - p_viscous
    
    # Calculate acceleration terms
    R_ddot = (delta_p / (rho_l * R)) - (3 * dR_dt * dR_dt) / (2 * R)
    
    return dR_dt, R_ddot

def solve_bubble_dynamics(R0, dR0_dt, t_span):
    """
    Solve the Rayleigh-Plesset equation for sodium with optimized computation
    """
    print("\nInitializing bubble dynamics calculation...")
    sleep(0.5)
    
    # Sodium properties at 0.5 bar
    T_sat = 881.0 + 273  # Saturation temperature at 0.5 bar [K]
    superheat = 340.0  # Specified superheat [K]
    T_inf = T_sat + superheat
    
    # Get initial physical properties based on T_inf
    rho_l = get_liquid_density(T_inf)  # Initial liquid density [kg/m³]
    rho_v = get_vapor_density(T_inf)  # Initial vapor density [kg/m³]
    k = 87.5  # Thermal conductivity [W/(m·K)]
    D = 8.3e-5  # Thermal diffusivity [m²/s]
    p_inf = 0.5e5  # Ambient pressure [Pa]
    sigma = get_surface_tension(T_inf)  # Initial surface tension [N/m]
    L = get_latent_heat(T_inf)  # Initial latent heat
    
    params = (rho_l, k, D, L, p_inf, sigma, T_inf, rho_v)
    
    # Initialize arrays
    n_points = len(t_span)
    R = np.zeros(n_points)
    dR_dt = np.zeros(n_points)
    T_s_history = np.zeros(n_points)
    
    # Set initial conditions
    R[0] = R0 * (1 + 1e-8)  # Add tiny perturbation to initial radius
    dR_dt[0] = dR0_dt
    
    # Initialize history arrays
    R_history = np.array([R0])
    t_history = np.array([t_span[0]])
    
    print("\nSolving Rayleigh-Plesset equation...")
    progress_steps = 10  # Number of progress marks to show
    last_percentage = -1
    
    for i in range(1, n_points):
        t = t_span[i]
        dt = t - t_span[i-1]
        
        # Show static progress bar
        current_percentage = int((i / n_points) * progress_steps)
        if current_percentage != last_percentage:
            progress = "=" * current_percentage + ">" + "." * (progress_steps - current_percentage - 1)
            print(f"\rProgress: [{progress}] {int((i / n_points) * 100)}%", end="", flush=True)
            last_percentage = current_percentage
            
            # Print current state on the same line
            if i % 100 == 0:
                print(f" | t={t:.2e}s R={R[i-1]:.2e}m v={dR_dt[i-1]:.2e}m/s", end="", flush=True)
        
        # Calculate surface temperature
        T_s = calculate_Ts_optimized(t, t_history, R_history, T_inf, k, D, rho_v, superheat)
        T_s_history[i-1] = T_s
        
        # Update all temperature-dependent properties
        rho_l = get_liquid_density(T_s)
        rho_v = get_vapor_density(T_s)
        L = get_latent_heat(T_s)
        sigma = get_surface_tension(T_s)  # Update surface tension based on current temperature
        
        params = (rho_l, k, D, L, p_inf, sigma, T_inf, rho_v)
        
        # Initialize current state variables
        R_current = max(R[i-1], 1e-12)  # Ensure minimum radius
        dR_dt_current = dR_dt[i-1]
        
        # Get initial acceleration for adaptive timestepping
        _, R_ddot = rayleigh_plesset_optimized(R_current, dR_dt_current, t, T_s, params)
        
        # Adaptive sub-stepping with improved numerical stability
        try:
            # Calculate substeps based on both velocity and radius changes
            vel_factor = abs(dR_dt_current) * dt / (0.01 * R_current)
            acc_factor = abs(R_ddot) * dt * dt / (0.01 * R_current)
            substep_factor = max(vel_factor, acc_factor)
            
            # Handle potential NaN or infinity
            if np.isnan(substep_factor) or np.isinf(substep_factor):
                n_substeps = 1000  # Default to high resolution if calculation fails
            else:
                n_substeps = max(1, min(5000, int(np.ceil(substep_factor))))
        except:
            # More robust fallback that avoids division by zero
            n_substeps = 1000 if R_current < 1e-9 else 200  # More substeps for very small bubbles
        
        dt_sub = dt / n_substeps
        
        # Sub-step integration with improved stability
        for _ in range(n_substeps):
            # Ensure R_current is positive and finite
            if not (np.isfinite(R_current) and R_current > 0):
                R_current = 1e-12
                dR_dt_current = 0.0
            
            try:
                dR_dt_new, R_ddot = rayleigh_plesset_optimized(
                    R_current, dR_dt_current, t, T_s, params
                )
                
                # Handle any NaN or infinite values
                if not np.isfinite(dR_dt_new) or not np.isfinite(R_ddot):
                    dR_dt_new = dR_dt_current
                    R_ddot = 0.0
                
                # Update with velocity limiting and stability checks
                R_new = max(1e-12, R_current + dt_sub * np.clip(dR_dt_new, -1e6, 1e6))
                dR_dt_current = np.clip(dR_dt_new + dt_sub * R_ddot, -1e6, 1e6)
                
                R_current = R_new
            except:
                # If any calculation fails, maintain previous values with small changes
                R_current = max(1e-12, R_current)
                dR_dt_current = 0.0
        
        # Ensure final values are physically meaningful
        R_current = max(1e-12, min(1e-3, R_current))  # Limit radius between 1 picometer and 1 millimeter
        dR_dt_current = np.clip(dR_dt_current, -1e6, 1e6)  # Limit velocity to ±1 km/s
        
        # Update state arrays
        R[i] = R_current
        dR_dt[i] = dR_dt_current
        
        # Update history arrays
        R_history = np.append(R_history, R_current)
        t_history = np.append(t_history, t)
    
    # Calculate final temperature point
    T_s_history[-1] = calculate_Ts_optimized(t_span[-1], t_history, R_history,
                                           T_inf, k, D, rho_v, superheat)
    
    print("\nCalculation complete!")
    return t_span, R, dR_dt, T_s_history

if __name__ == "__main__":
    try:
        print("Starting bubble dynamics simulation...")
        print("-" * 50, flush=True)
        
        # Initial conditions
        R0 = 1e-6  # Initial radius 1 micrometer [m]
        dR0_dt = 0.0  # Initial velocity [m/s]
        print(f"Initial conditions: R0 = {R0:.2e} m, dR0_dt = {dR0_dt:.2e} m/s", flush=True)
        
        # Temperature initialization
        T_sat = 881.0 + 273  # Saturation temperature at 0.5 bar [K]
        superheat = 340.0  # Specified superheat [K]
        T_inf = T_sat + superheat  # Initial temperature [K]
        print(f"Temperature parameters: T_sat = {T_sat:.2f} K, T_inf = {T_inf:.2f} K", flush=True)
        
        # Time span with extended bounds
        t_start = 1e-10  # Start time [s]
        t_end = 1.0      # End time [s]
        n_points = 1000  # Number of time points
        t_span = np.logspace(np.log10(t_start), np.log10(t_end), n_points)
        print(f"Time span: {t_start:.2e} to {t_end:.2e} seconds", flush=True)
        
        print("\nCalling solve_bubble_dynamics...")
        t, R, dR_dt, T_s_history = solve_bubble_dynamics(R0, dR0_dt, t_span)
        print("Finished solve_bubble_dynamics")
        
        if np.any(np.isnan(R)):
            print("Warning: NaN values in R array")
        if np.any(np.isnan(dR_dt)):
            print("Warning: NaN values in dR_dt array")
        if np.any(np.isnan(T_s_history)):
            print("Warning: NaN values in T_s_history array")
        
        # Save the plots
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Bubble Wall Velocity (in cm/s)
        plt.subplot(2, 1, 1)
        plt.semilogx(t, dR_dt * 100, 'b-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Bubble Wall Velocity (cm/s)')
        plt.title('Bubble Wall Velocity vs Time')
        plt.grid(True)
        plt.ylim(-100, 10000)
        
        # Plot 2: Surface Temperature
        plt.subplot(2, 1, 2)
        plt.semilogx(t, T_s_history, 'r-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Surface Temperature (K)')
        plt.title('Surface Temperature vs Time')
        plt.grid(True)
        plt.ylim(1100, 1500)
        
        plt.tight_layout()
        plt.savefig('bubble_dynamics_results.png')
        print("\nPlots saved as 'bubble_dynamics_results.png'")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()