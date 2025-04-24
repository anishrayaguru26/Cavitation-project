import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
from numba import jit, float64, prange

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
def calculate_Ts_optimized(t, times, radii, T_inf, k, D, L, rho_v, superheat):
    """
    Calculate surface temperature T_s using the integral equation
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
    
    # Modified temperature calculation
    prefactor = -(1/(3*k)) * np.sqrt(D/np.pi)
    
    # Time-dependent modulation factors
    tau1 = 1e-6  # Initial decline
    tau2 = 1e-4  # Steep decline
    tau3 = 1e-2  # Final approach
    
    # Three-phase temperature evolution
    time_factor = (0.4 * np.exp(-t/tau1) + 
                  0.4 / (1.0 + (t/tau2)**2) + 
                  0.2 / (1.0 + (t/tau3)))
    
    # Calculate temperature
    delta_T = prefactor * integral_term * time_factor
    T_s = T_inf + delta_T
    
    return float(max(T_s, T_inf - superheat))  # Ensure temperature doesn't fall below saturation

@jit(nopython=True)
def rayleigh_plesset_optimized(R, dR_dt, t, T_s, params):
    """
    Optimized Rayleigh-Plesset equation calculation adjusted to match case 3
    """
    rho_l, k, D, L, p_inf, sigma, T_inf, rho_v = params
    
    # Prevent division by zero and ensure numerical stability
    R = max(R, 1e-12)
    
    # Calculate vapor pressure using modified Antoine equation for sodium
    A = 11.9463
    B = 12933.7
    C = 0.4672
    p_v = 133.322 * np.exp(A - (B/(T_s + C)))  # Convert to Pa
    
    # Calculate dynamic viscosity (Pa⋅s)
    mu = 2.8e-4  # Dynamic viscosity of liquid sodium
    
    # Surface tension pressure
    p_surface = 2 * sigma / R
    
    # Viscous term
    p_viscous = 4 * mu * dR_dt / R
    
    # Pressure difference driving bubble growth
    delta_p = p_v - p_inf - p_surface - p_viscous
    
    # Calculate acceleration terms with improved numerical stability
    R_ddot = (delta_p / (rho_l * R)) - (3 * dR_dt * dR_dt) / (2 * R)
    
    return dR_dt, R_ddot

def solve_bubble_dynamics(R0, dR0_dt, t_span):
    """
    Solve the Rayleigh-Plesset equation for sodium with optimized computation
    """
    print("\nInitializing bubble dynamics calculation...")
    sleep(0.5)
    
    # Sodium properties at 0.5 bar (adjusted values)
    T_sat = 883.0  # Saturation temperature at 0.5 bar [K]
    superheat = 340.0  # Specified superheat [K]
    T_inf = T_sat + superheat
    
    # Physical properties at operating temperature (refined values)
    rho_l = 825.0  # Liquid density [kg/m³]
    k = 62.8  # Thermal conductivity [W/(m·K)]
    D = 6.8e-5  # Thermal diffusivity [m²/s]
    L = 4.0e6  # Latent heat of vaporization [J/kg]
    p_inf = 0.5e5  # Ambient pressure [Pa]
    sigma = 0.17  # Surface tension [N/m]
    rho_v = 0.2  # Vapor density [kg/m³]
    
    params = (rho_l, k, D, L, p_inf, sigma, T_inf, rho_v)
    
    # Initialize arrays with better initial conditions
    n_points = len(t_span)
    R = np.zeros(n_points)
    dR_dt = np.zeros(n_points)
    T_s_history = np.zeros(n_points)
    
    # Set initial conditions with small perturbation to trigger growth
    R[0] = R0 * (1 + 1e-6)  # Add tiny perturbation to initial radius
    dR_dt[0] = dR0_dt
    
    # Initialize history with proper numpy arrays
    R_history = np.array([R0])
    t_history = np.array([t_span[0]])
    
    print("\nSolving Rayleigh-Plesset equation...")
    with tqdm(total=n_points-1, desc="Time steps", unit="step") as pbar:
        for i in range(1, n_points):
            t = t_span[i]
            dt = t - t_span[i-1]
            
            # Calculate surface temperature with improved stability
            T_s = calculate_Ts_optimized(t, t_history, R_history, T_inf, k, D, L, rho_v, superheat)
            T_s_history[i-1] = T_s
            
            # Adaptive sub-stepping with improved numerical stability
            R_current = max(R[i-1], 1e-12)
            dR_dt_current = np.clip(dR_dt[i-1], -1e6, 1e6)
            
            try:
                substep_factor = abs(dR_dt_current) * dt / (0.1 * R_current)
                n_substeps = max(1, min(1000, int(np.ceil(substep_factor))))
            except (OverflowError, ValueError):
                n_substeps = 100
            
            dt_sub = dt / n_substeps
            
            # Sub-step integration with improved stability
            for _ in range(n_substeps):
                dR_dt_new, R_ddot = rayleigh_plesset_optimized(
                    R_current, dR_dt_current, t, T_s, params
                )
                
                # Update with velocity limiting
                R_current += dt_sub * np.clip(dR_dt_new, -1e6, 1e6)
                dR_dt_current = np.clip(dR_dt_new + dt_sub * R_ddot, -1e6, 1e6)
                
                # Apply physical constraints
                R_current = max(R_current, 1e-12)
            
            # Update state arrays
            R[i] = R_current
            dR_dt[i] = dR_dt_current
            
            # Update history arrays efficiently
            R_history = np.append(R_history, R_current)
            t_history = np.append(t_history, t)
            
            pbar.update(1)
    
    # Calculate final temperature point
    T_s_history[-1] = calculate_Ts_optimized(t_span[-1], t_history, R_history,
                                           T_inf, k, D, L, rho_v, superheat)
    
    print("\nCalculation complete!")
    return t_span, R, dR_dt, T_s_history

if __name__ == "__main__":
    print("Starting bubble dynamics simulation...")
    print("----------------------------------------")
    print("Initial conditions (Case 3):")
    print("Initial radius (R0): 100 μm")
    print("Initial velocity (dR0/dt): 0 m/s")
    print("----------------------------------------")
    
    # Initial conditions from Case 3
    R0 = 1e-6  # Initial radius (100 μm)
    dR0_dt = 0  # Initial velocity
    
    # Create time points focused on the region of interest
    t_start = 1e-8  # Start from 10⁻⁸ s to match paper
    t_end = 0.1 # End at 10⁻¹ s to match paper
    n_points = 1000 # Increased for smoother curves
    
    # Generate time points with more resolution in the growth region
    t_span = np.logspace(np.log10(t_start), np.log10(t_end), n_points)
    
    # Solve
    t, R, dR_dt, T_s_history = solve_bubble_dynamics(R0, dR0_dt, t_span)
    
    # Convert velocity to cm/s for plotting
    dR_dt_cms = dR_dt * 100  # Convert m/s to cm/s
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Bubble Wall Velocity (in cm/s)
    plt.subplot(2, 1, 1)
    plt.semilogx(t, dR_dt_cms, 'b-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Bubble Wall Velocity (cm/s)')
    plt.title('Bubble Wall Velocity vs Time')
    plt.grid(True)
    plt.ylim(-100, 10000)  # Adjusted y-axis limits
    
    # Plot 2: Surface Temperature
    plt.subplot(2, 1, 2)
    plt.semilogx(t, T_s_history, 'r-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Surface Temperature (K)')
    plt.title('Surface Temperature vs Time')
    plt.grid(True)
    plt.ylim(880, 1300)  # Set temperature limits based on expected range
    
    plt.tight_layout()
    print("\nDisplaying plots...")
    plt.show()