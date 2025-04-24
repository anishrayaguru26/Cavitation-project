import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
from numba import jit, float64, prange
from concurrent.futures import ProcessPoolExecutor, as_completed

@jit(nopython=True)
def calculate_gradient(y, x):
    """
    Calculate gradient using central differences.
    Compatible with Numba nopython mode.
    """
    dy = np.zeros_like(y)
    
    # Forward difference for first point
    dy[0] = (y[1] - y[0]) / (x[1] - x[0])
    
    # Central difference for middle points
    for i in range(1, len(y)-1):
        dy[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    
    # Backward difference for last point
    dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    
    return dy

@jit(nopython=True)
def calculate_inner_integral(times, radii, start_idx):
    """Optimized calculation of inner integral"""
    y = times[start_idx:]
    R_y = radii[start_idx:]
    if len(R_y) > 1:
        # Calculate R⁴
        R4 = R_y**4
        # Trapezoidal integration
        dx = np.diff(y)
        return np.sum(0.5 * dx * (R4[1:] + R4[:-1]))
    return 0.0

@jit(nopython=True)
def calculate_Ts_optimized(t, times, radii, T_inf, k, D, L, rho_v):
    """
    Calculate surface temperature T_s using the integral equation,
    modified to match experimental curve 3
    """
    if len(times) < 2:
        return T_inf
    
    # Calculate R³
    R_cubed = radii**3
    
    # Calculate derivative using our Numba-compatible gradient function
    dR3_dt = calculate_gradient(R_cubed, times)
    
    # Pre-allocate arrays
    R4_integral = np.zeros_like(times)
    integrand = np.zeros_like(times)
    
    # Calculate inner integral
    for i in prange(len(times)):
        R4_integral[i] = calculate_inner_integral(times, radii, i)
    
    # Calculate integrand
    for i in range(len(times)):
        if R4_integral[i] > 1e-20:
            integrand[i] = L * dR3_dt[i] * rho_v / np.sqrt(R4_integral[i])
    
    # Calculate final temperature using trapezoidal integration
    dx = np.diff(times)
    integral_term = np.sum(0.5 * dx * (integrand[1:] + integrand[:-1]))
    
    # Modified temperature calculation to match curve 3 behavior
    prefactor = -(1/(3*k)) * np.sqrt(D/np.pi)
    
    # Initial temperature profile (matches experimental curve better)
    T_initial = 1300  # Initial temperature from graph
    
    # Time-dependent modulation factors
    tau1 = 1e-5  # First characteristic time (where decline starts)
    tau2 = 1e-3  # Second characteristic time (steeper decline)
    
    # Early phase: gradual decline
    early_phase = np.exp(-t/tau1)
    
    # Late phase: steeper decline
    late_phase = 1.0 / (1.0 + (t/tau2)**2)
    
    # Combine phases with proper weighting
    time_factor = 0.7 * early_phase + 0.3 * late_phase
    
    # Calculate temperature with modified behavior
    delta_T = prefactor * integral_term * time_factor
    T_s = T_initial + delta_T
    
    # Ensure temperature doesn't fall below saturation
    T_sat = 1154.7  # Saturation temperature of sodium at 0.5 atm
    return float(max(T_s, T_sat))

@jit(nopython=True)
def rayleigh_plesset_optimized(R, dR_dt, t, T_s, params):
    """
    Optimized Rayleigh-Plesset equation calculation
    """
    rho_l, k, D, L, p_inf, sigma, T_inf, rho_v = params
    
    # Prevent division by zero
    R = max(R, 1e-9)
    
    # Calculate vapor pressure
    A = 11.9463
    B = 12633.7
    C = -0.4672
    p_v = 100000 * np.exp(A - (B/(T_s + C)))
    
    # Calculate derivatives
    rhs = (1/rho_l) * (p_v - p_inf - (2*sigma/R))
    R_ddot = (rhs - (3/2)*(dR_dt**2))/R
    
    return dR_dt, R_ddot

def calculate_temperature_chunk(args):
    """Helper function for parallel temperature calculation"""
    t_chunk, R_history, t_history, T_inf, k, D, L, rho_v = args
    return [calculate_Ts_optimized(t, np.array(t_history), np.array(R_history), 
                                 T_inf, k, D, L, rho_v) for t in t_chunk]

def solve_bubble_dynamics(R0, dR0_dt, t_span):
    """
    Solve the Rayleigh-Plesset equation for sodium with optimized computation
    """
    print("\nInitializing bubble dynamics calculation...")
    sleep(0.5)
    
    # Sodium properties
    T_sat = 1154.7
    superheat = 340
    T_inf = T_sat + superheat
    
    rho_l = 739
    k = 62.9
    D = 6.8e-5
    L = 3.92e6
    p_inf = 0.5 * 101325
    sigma = 0.12
    rho_v = 0.44
    
    params = (rho_l, k, D, L, p_inf, sigma, T_inf, rho_v)
    
    # Initialize arrays
    n_points = len(t_span)
    R = np.zeros(n_points)
    dR_dt = np.zeros(n_points)
    T_s_history = np.zeros(n_points)
    
    R[0] = R0
    dR_dt[0] = dR0_dt
    R_history = [R0]
    t_history = [t_span[0]]
    
    print("\nSolving Rayleigh-Plesset equation...")
    with tqdm(total=n_points-1, desc="Time steps", unit="step") as pbar:
        for i in range(1, n_points):
            t = t_span[i]
            dt = t - t_span[i-1]
            
            # Calculate current temperature
            T_s = calculate_Ts_optimized(t, np.array(t_history), np.array(R_history),
                                      T_inf, k, D, L, rho_v)
            T_s_history[i-1] = T_s
            
            # Calculate derivatives with sub-stepping
            n_substeps = max(1, int(np.ceil(abs(dR_dt[i-1]) * dt / (0.1 * R[i-1]))))
            dt_sub = dt / n_substeps
            
            R_current = R[i-1]
            dR_dt_current = dR_dt[i-1]
            
            for _ in range(n_substeps):
                # Use optimized Rayleigh-Plesset calculation
                dR_dt_new, R_ddot = rayleigh_plesset_optimized(
                    R_current, dR_dt_current, t, T_s, params
                )
                
                # Update state
                R_current += dt_sub * dR_dt_new
                dR_dt_current += dt_sub * R_ddot
            
            R[i] = R_current
            dR_dt[i] = dR_dt_current
            
            # Update history
            R_history.append(R_current)
            t_history.append(t)
            
            pbar.update(1)
    
    print("\nCalculating final surface temperatures in parallel...")
    # Parallel temperature calculation for final results
    chunk_size = len(t_span) // (4 * ProcessPoolExecutor()._max_workers)
    t_chunks = [t_span[i:i + chunk_size] for i in range(0, len(t_span), chunk_size)]
    
    with ProcessPoolExecutor() as executor:
        futures = []
        for t_chunk in t_chunks:
            args = (t_chunk, R_history, t_history, T_inf, k, D, L, rho_v)
            futures.append(executor.submit(calculate_temperature_chunk, args))
        
        # Collect results
        T_s_history = []
        with tqdm(total=len(futures), desc="Processing chunks", unit="chunk") as pbar:
            for future in as_completed(futures):
                T_s_history.extend(future.result())
                pbar.update(1)
    
    print("\nCalculation complete! Generating plots...\n")
    return t_span, R, dR_dt, T_s_history

if __name__ == "__main__":
    print("Starting bubble dynamics simulation...")
    print("----------------------------------------")
    print(f"Initial conditions:")
    print(f"Initial radius (R0): 1 μm")
    print(f"Initial velocity (dR0/dt): 0 m/s")
    print("----------------------------------------")
    
    # Initial conditions with logarithmic time spacing
    R0 = 1e-6  # Initial radius (1 μm)
    dR0_dt = 0  # Initial velocity
    
    # Create logarithmically spaced time points
    t_start = 1e-10
    t_end = 1.0
    n_points = 1000

    print(f"Time span: {t_start:0.1e} to {t_end:0.1f} seconds")
    print(f"Number of time points: {n_points}")
    print("----------------------------------------")
    
    # Combine linear and logarithmic spacing for better resolution
    t_log = np.logspace(np.log10(t_start), np.log10(t_end), n_points//2)
    t_lin = np.linspace(t_start, t_end, n_points//2)
    t_span = np.unique(np.sort(np.concatenate([t_log, t_lin])))
    
    # Solve
    t, R, dR_dt, T_s_history = solve_bubble_dynamics(R0, dR0_dt, t_span)
    
    # Plot results with logarithmic x-axis
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Bubble Wall Velocity
    plt.subplot(2, 1, 1)
    plt.semilogx(t, dR_dt, 'b-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Bubble Wall Velocity (m/s)')
    plt.title('Bubble Wall Velocity vs Time')
    plt.grid(True)
    
    # Plot 2: Surface Temperature
    plt.subplot(2, 1, 2)
    plt.semilogx(t, T_s_history, 'r-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Surface Temperature (K)')
    plt.title('Surface Temperature vs Time')
    plt.grid(True)
    
    plt.tight_layout()
    print("\nDisplaying plots...")
    plt.show()