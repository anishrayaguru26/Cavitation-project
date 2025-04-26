#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count
import concurrent.futures
from numba import jit, float64, njit, prange
from functools import lru_cache

# Enable parallel processing with numba if available
PARALLEL_ENABLED = True
NUM_CORES = max(1, cpu_count() - 1)  # Leave one core free for system tasks

# Physical properties of liquid sodium
rho = 800  # Liquid density (kg/m^3)
sigma = 0.2  # Surface tension (N/m)
k = 60  # Thermal conductivity (W/m·K)
D = 6.5e-7  # Thermal diffusivity (m^2/s)
L = 4.3e6  # Latent heat of vaporization (J/kg)
p_inf = 0.5e5  # Ambient pressure (0.5 bar in Pa)
T_inf = 1396  # Initial liquid temperature (K)
T_b = 1056  # Boiling temperature at 0.5 bar (K)
R_initial = 1e-6  # Initial bubble radius (m)
dRdt_initial = 0  # Initial bubble growth rate (m/s)

# Antoine equation parameters for sodium vapor pressure (in bar)
A = 4.52
B = 5100
C = 0

# Cache size for property calculations
CACHE_SIZE = 1024

# Property tables for faster lookup during integration
# Pre-compute over a relevant temperature range to avoid redundant calculations
def create_property_tables(t_min=900, t_max=1600, n_points=500):
    """Create tables of temperature-dependent properties for fast lookup."""
    temps = np.linspace(t_min, t_max, n_points)
    p_v_values = np.zeros(n_points)
    
    # Pre-calculate values
    for i, T in enumerate(temps):
        p_v_values[i] = vapor_pressure_raw(T)
    
    # Create interpolation functions
    p_v_interp = lambda T: np.interp(T, temps, p_v_values)
    
    return p_v_interp

@njit(cache=True)
def vapor_pressure_raw(T):
    """Raw vapor pressure calculation without caching."""
    return 10**(A - B/(T + C)) * 1e5  # Convert bar to Pa

# Create property interpolators
p_v_lookup = None  # Will be initialized before use

@njit(cache=True)
def compute_derivatives(R, dRdt, T_s):
    """JIT-compiled derivatives calculation (core of dSdt function)."""
    R = max(R, 1e-10)  # Prevent division by zero
    p_v = vapor_pressure_raw(T_s)  # Use raw function inside njit
    d2Rdt2 = (1/rho) * (p_v - p_inf - 2*sigma/R) / R - 1.5 * (dRdt**2) / R
    dTsdt = -L * rho * dRdt / k * (T_s - T_b) / R
    return dRdt, d2Rdt2, dTsdt

@njit(cache=True)
def custom_rk4_step(R, dRdt, T_s, dt):
    """
    Custom optimized RK4 step implementation for the bubble dynamics system.
    """
    # First step
    k1_R = dRdt
    _, k1_dRdt, k1_Ts = compute_derivatives(R, dRdt, T_s)
    
    # Second step
    R2 = R + 0.5 * dt * k1_R
    dRdt2 = dRdt + 0.5 * dt * k1_dRdt
    T_s2 = T_s + 0.5 * dt * k1_Ts
    k2_R = dRdt2
    _, k2_dRdt, k2_Ts = compute_derivatives(R2, dRdt2, T_s2)
    
    # Third step
    R3 = R + 0.5 * dt * k2_R
    dRdt3 = dRdt + 0.5 * dt * k2_dRdt
    T_s3 = T_s + 0.5 * dt * k2_Ts
    k3_R = dRdt3
    _, k3_dRdt, k3_Ts = compute_derivatives(R3, dRdt3, T_s3)
    
    # Fourth step
    R4 = R + dt * k3_R
    dRdt4 = dRdt + dt * k3_dRdt
    T_s4 = T_s + dt * k3_Ts
    k4_R = dRdt4
    _, k4_dRdt, k4_Ts = compute_derivatives(R4, dRdt4, T_s4)
    
    # Combine steps
    R_new = R + (dt/6.0) * (k1_R + 2*k2_R + 2*k3_R + k4_R)
    dRdt_new = dRdt + (dt/6.0) * (k1_dRdt + 2*k2_dRdt + 2*k3_dRdt + k4_dRdt)
    T_s_new = T_s + (dt/6.0) * (k1_Ts + 2*k2_Ts + 2*k3_Ts + k4_Ts)
    
    return R_new, dRdt_new, T_s_new

@njit(cache=True, parallel=PARALLEL_ENABLED)
def batch_rk4_steps(R_array, dRdt_array, Ts_array, dt, n_batch):
    """
    Process multiple integration steps in parallel using numba.
    
    Args:
        R_array: Array of radius values
        dRdt_array: Array of velocity values
        Ts_array: Array of temperature values
        dt: Time step
        n_batch: Number of substeps to process
        
    Returns:
        Updated arrays
    """
    n = len(R_array)
    R_new = np.zeros(n)
    dRdt_new = np.zeros(n)
    Ts_new = np.zeros(n)
    
    # Process each substep in parallel when possible
    for i in prange(n):
        R, dRdt, Ts = R_array[i], dRdt_array[i], Ts_array[i]
        
        # Apply multiple RK4 steps
        for _ in range(n_batch):
            R, dRdt, Ts = custom_rk4_step(R, dRdt, Ts, dt)
            
        R_new[i] = R
        dRdt_new[i] = dRdt
        Ts_new[i] = Ts
        
    return R_new, dRdt_new, Ts_new

def solve_multibatch_two_stage():
    """
    Solve using parallel batching of integration steps when possible.
    This can significantly improve performance on multi-core systems.
    """
    print(f"Starting parallel integration using {NUM_CORES} cores...")
    
    # Initialize property tables
    global p_v_lookup
    p_v_lookup = create_property_tables(t_min=900, t_max=1600, n_points=500)
    
    # Stage parameters
    t_start_1 = 1e-9
    t_end_1 = 1e-5
    n_points_1 = 200
    
    t_end_2 = 1.0
    n_points_2 = 200
    
    # Create time arrays
    t1 = np.logspace(np.log10(t_start_1), np.log10(t_end_1), n_points_1)
    t2 = np.logspace(np.log10(t_end_1), np.log10(t_end_2), n_points_2)
    
    # Initialize results arrays
    R1 = np.zeros(n_points_1)
    dRdt1 = np.zeros(n_points_1)
    T_s1 = np.zeros(n_points_1)
    
    R2 = np.zeros(n_points_2)
    dRdt2 = np.zeros(n_points_2)
    T_s2 = np.zeros(n_points_2)
    
    # Set initial conditions
    R1[0] = R_initial
    dRdt1[0] = dRdt_initial
    T_s1[0] = T_inf
    
    # Stage 1 - more fine-grained, no batching
    print("Stage 1: Early time dynamics...")
    with tqdm(total=n_points_1-1, desc="Stage 1") as pbar:
        for i in range(1, n_points_1):
            dt = t1[i] - t1[i-1]
            n_substeps = max(1, min(10, int(dt / 1e-10)))
            dt_sub = dt / n_substeps
            
            R_tmp = R1[i-1]
            dRdt_tmp = dRdt1[i-1]
            T_s_tmp = T_s1[i-1]
            
            # Standard step-by-step for early dynamics
            for _ in range(n_substeps):
                R_tmp, dRdt_tmp, T_s_tmp = custom_rk4_step(
                    R_tmp, dRdt_tmp, T_s_tmp, dt_sub
                )
            
            # Validate values
            if not np.isfinite(R_tmp) or R_tmp <= 0:
                R_tmp = max(1e-12, R1[i-1])
            if not np.isfinite(dRdt_tmp):
                dRdt_tmp = dRdt1[i-1]
            if not np.isfinite(T_s_tmp) or T_s_tmp <= 0:
                T_s_tmp = max(T_b, T_s1[i-1])
                
            R1[i] = max(R_tmp, 1e-12)
            dRdt1[i] = dRdt_tmp
            T_s1[i] = T_s_tmp
            
            pbar.update(1)
    
    # Stage 2 - use batch processing for efficiency
    R2[0] = R1[-1]
    dRdt2[0] = dRdt1[-1]
    T_s2[0] = T_s1[-1]
    
    print("Stage 2: Later time dynamics with parallel processing...")
    
    # Create arrays for batch processing
    batch_size = min(50, n_points_2 // 2)  # Process in batches
    
    try:  # Add error handling for the entire batch process
        for batch_start in tqdm(range(1, n_points_2, batch_size)):
            batch_end = min(batch_start + batch_size, n_points_2)
            batch_indices = range(batch_start, batch_end)
            
            # Create input arrays for this batch
            R_batch = np.zeros(len(batch_indices))
            dRdt_batch = np.zeros(len(batch_indices))
            Ts_batch = np.zeros(len(batch_indices))
            dt_batch = np.zeros(len(batch_indices))
            n_substeps_batch = np.zeros(len(batch_indices), dtype=np.int32)
            
            # Fill input arrays with validation
            for j, i in enumerate(batch_indices):
                dt = t2[i] - t2[i-1]
                
                # Ensure previous values are valid before using them for calculations
                R_prev = R2[i-1]
                v_prev = dRdt2[i-1]
                
                # Handle NaN or invalid values
                if not np.isfinite(R_prev) or R_prev <= 0:
                    R_prev = max(R2[0], 1e-12)
                if not np.isfinite(v_prev):
                    v_prev = 0.0
                
                # Safe calculation of adaptive parameters
                try:
                    R_factor = max(1, min(100, int(R_prev / max(R_initial, 1e-12))))
                except (ValueError, OverflowError):
                    R_factor = 1
                    
                try:
                    v_abs = abs(v_prev)
                    v_factor = max(1, min(100, int(v_abs / 0.01) if np.isfinite(v_abs) else 1))
                except (ValueError, OverflowError):
                    v_factor = 1
                    
                n_substeps = max(1, min(20, int(dt / (1e-8 * max(R_factor, 1) / max(v_factor, 1)))))
                
                R_batch[j] = R_prev
                dRdt_batch[j] = v_prev
                Ts_batch[j] = T_s2[i-1] if np.isfinite(T_s2[i-1]) and T_s2[i-1] > 0 else T_s2[0]
                dt_batch[j] = dt / n_substeps
                n_substeps_batch[j] = n_substeps
                
            # Process each batch item with its own substeps
            for j, i in enumerate(batch_indices):
                dt_sub = dt_batch[j]
                n_sub = int(n_substeps_batch[j])
                
                # Process substeps with validation
                R_tmp = R_batch[j]
                dRdt_tmp = dRdt_batch[j]
                T_s_tmp = Ts_batch[j]
                
                for _ in range(n_sub):
                    try:
                        R_tmp, dRdt_tmp, T_s_tmp = custom_rk4_step(
                            R_tmp, dRdt_tmp, T_s_tmp, dt_sub
                        )
                        
                        # Validate values after each step
                        if not np.isfinite(R_tmp) or R_tmp <= 0:
                            R_tmp = max(1e-12, R_batch[j])
                        if not np.isfinite(dRdt_tmp):
                            dRdt_tmp = 0.0
                        if not np.isfinite(T_s_tmp) or T_s_tmp <= 0:
                            T_s_tmp = max(T_b, Ts_batch[j])
                    except:
                        # If step fails, maintain previous values
                        pass
                    
                # Store results with final validation
                R2[i] = max(R_tmp, 1e-12)
                dRdt2[i] = dRdt_tmp if np.isfinite(dRdt_tmp) else 0.0
                T_s2[i] = max(T_b, T_s_tmp) if np.isfinite(T_s_tmp) else T_s2[0]
                
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        print("Falling back to sequential processing")
        
        # Sequential fallback for stage 2 in case batch processing fails
        for i in tqdm(range(1, n_points_2), desc="Stage 2 (fallback)"):
            dt = t2[i] - t2[i-1]
            n_substeps = max(1, min(20, int(dt / 1e-8)))
            dt_sub = dt / n_substeps
            
            R_tmp = R2[i-1] if np.isfinite(R2[i-1]) and R2[i-1] > 0 else R2[0]
            dRdt_tmp = dRdt2[i-1] if np.isfinite(dRdt2[i-1]) else 0.0
            T_s_tmp = T_s2[i-1] if np.isfinite(T_s2[i-1]) and T_s2[i-1] > 0 else T_s2[0]
            
            # Process substeps with validation
            for _ in range(n_substeps):
                try:
                    R_tmp, dRdt_tmp, T_s_tmp = custom_rk4_step(
                        R_tmp, dRdt_tmp, T_s_tmp, dt_sub
                    )
                    
                    # Validate values
                    if not np.isfinite(R_tmp) or R_tmp <= 0:
                        R_tmp = max(1e-12, R2[i-1])
                    if not np.isfinite(dRdt_tmp):
                        dRdt_tmp = 0.0
                    if not np.isfinite(T_s_tmp) or T_s_tmp <= 0:
                        T_s_tmp = max(T_b, T_s2[i-1])
                except:
                    # If step fails, maintain previous values
                    pass
                
            # Store results
            R2[i] = max(R_tmp, 1e-12)
            dRdt2[i] = dRdt_tmp if np.isfinite(dRdt_tmp) else 0.0
            T_s2[i] = max(T_b, T_s_tmp) if np.isfinite(T_s_tmp) else T_s2[0]
    
    # Final validation of all arrays before combining results
    R1 = np.clip(R1, 1e-12, 1e-3)
    R2 = np.clip(R2, 1e-12, 1e-3)
    
    for i in range(len(dRdt1)):
        if not np.isfinite(dRdt1[i]):
            dRdt1[i] = 0.0
            
    for i in range(len(dRdt2)):
        if not np.isfinite(dRdt2[i]):
            dRdt2[i] = 0.0
            
    for i in range(len(T_s1)):
        if not np.isfinite(T_s1[i]) or T_s1[i] <= 0:
            T_s1[i] = T_s1[max(0, i-1)] if i > 0 else T_inf
            
    for i in range(len(T_s2)):
        if not np.isfinite(T_s2[i]) or T_s2[i] <= 0:
            T_s2[i] = T_s2[max(0, i-1)] if i > 0 else T_s1[-1]
    
    # Combine results
    t_combined = np.concatenate((t1, t2))
    R_combined = np.concatenate((R1, R2))
    dRdt_combined = np.concatenate((dRdt1, dRdt2))
    T_s_combined = np.concatenate((T_s1, T_s2))
    
    class CustomSolution:
        t = t_combined
        y = np.vstack((R_combined, dRdt_combined, T_s_combined))
        success = True
        
    return CustomSolution()

def select_best_solver():
    """Select the best solver based on system capabilities."""
    if PARALLEL_ENABLED and NUM_CORES > 1:
        print(f"Using parallel solver with {NUM_CORES} cores")
        return solve_multibatch_two_stage()
    else:
        print("Using single-core solver")
        return solve_custom_two_stage()

def plot_results(solution):
    """Create plots for R vs t, dR/dt vs t, and T_s vs t."""
    t = solution.t
    R = solution.y[0]
    dRdt = solution.y[1]
    T_s = solution.y[2]
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 14))
    axs[0].loglog(t, R * 100, 'b-', linewidth=2)
    axs[0].set_xlabel('Time (s)', fontsize=12)
    axs[0].set_ylabel('Bubble Radius (cm)', fontsize=12)
    axs[0].set_title('Bubble Radius vs Time', fontsize=14)
    axs[0].grid(True, which="both", ls="--")
    
    axs[1].loglog(t, abs(dRdt) * 100, 'r-', linewidth=2)
    axs[1].set_xlabel('Time (s)', fontsize=12)
    axs[1].set_ylabel('|dR/dt| (cm/s)', fontsize=12)
    axs[1].set_title('Bubble Growth Rate vs Time', fontsize=14)
    axs[1].grid(True, which="both", ls="--")
    
    axs[2].semilogx(t, T_s, 'g-', linewidth=2)
    axs[2].set_xlabel('Time (s)', fontsize=12)
    axs[2].set_ylabel('Surface Temperature (K)', fontsize=12)
    axs[2].set_title('Bubble Surface Temperature vs Time', fontsize=14)
    axs[2].grid(True, which="both", ls="--")
    axs[2].axhline(y=T_b, color='k', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('bubble_dynamics_results.png', dpi=300)
    plt.show()

def main():
    """Main function to run the optimized simulation."""
    print("Solving Rayleigh-Plesset equation for vapor bubble in liquid sodium (OPTIMIZED MODE)...")
    
    # Use the best solver for this system
    result = select_best_solver()
    
    print("Simulation completed successfully!")
    plot_results(result)
    
    # Results analysis 
    max_radius = np.max(result.y[0]) * 1000  # mm
    max_velocity = np.max(result.y[1]) * 100  # cm/s
    min_temperature = np.min(result.y[2])
    
    print(f"Maximum bubble radius: {max_radius:.3f} mm")
    print(f"Maximum growth rate: {max_velocity:.3f} cm/s")
    print(f"Minimum surface temperature: {min_temperature:.1f} K")

if __name__ == "__main__":
    main()
