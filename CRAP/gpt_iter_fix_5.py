#!/usr/bin/env python3
"""
Simplified bubble dynamics simulation with RK4 integrator
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit

# Physical properties of sodium
rho_l = 800.0      # Liquid density [kg/m³]
sigma = 0.2        # Surface tension [N/m]
k = 60.0           # Thermal conductivity [W/(m·K)]
D = 6.5e-7         # Thermal diffusivity [m²/s]
L = 4.3e6          # Latent heat of vaporization [J/kg]
p_inf = 0.5e5      # Ambient pressure (0.5 bar) [Pa]
T_inf = 1396.0     # Initial liquid temperature [K]
T_b = 1056.0       # Boiling temperature at 0.5 bar [K]
R_initial = 1e-6   # Initial bubble radius [m]
dRdt_initial = 0.0 # Initial velocity [m/s]

# Antoine equation parameters for sodium vapor pressure (in bar)
A = 4.52
B = 5100.0
C = 0.0

@njit
def vapor_pressure(T):
    """Calculate vapor pressure of sodium using Antoine equation"""
    return 10.0**(A - B/(T + C)) * 1e5  # Convert bar to Pa

@njit
def rayleigh_plesset_derivatives(R, dRdt, T_s):
    """Calculate derivatives for the Rayleigh-Plesset equation"""
    # Calculate vapor pressure based on surface temperature
    p_v = vapor_pressure(T_s)
    
    # Calculate second derivative (acceleration)
    R = max(R, 1e-12)  # Prevent division by zero
    d2Rdt2 = (1/rho_l) * (p_v - p_inf - 2*sigma/R) / R - 1.5 * (dRdt**2) / R
    
    # Temperature evolution based on energy balance
    dTsdt = -L * rho_l * dRdt / (k * max(R, 1e-12)) * (T_s - T_b)
    
    return dRdt, d2Rdt2, dTsdt

@njit
def rk4_step(R, dRdt, T_s, dt):
    """Fourth-order Runge-Kutta integrator for bubble dynamics"""
    # k1
    k1_dRdt, k1_d2Rdt2, k1_dTsdt = rayleigh_plesset_derivatives(R, dRdt, T_s)
    
    # k2
    R2 = R + 0.5 * dt * k1_dRdt
    dRdt2 = dRdt + 0.5 * dt * k1_d2Rdt2
    T_s2 = T_s + 0.5 * dt * k1_dTsdt
    k2_dRdt, k2_d2Rdt2, k2_dTsdt = rayleigh_plesset_derivatives(R2, dRdt2, T_s2)
    
    # k3
    R3 = R + 0.5 * dt * k2_dRdt
    dRdt3 = dRdt + 0.5 * dt * k2_d2Rdt2
    T_s3 = T_s + 0.5 * dt * k2_dTsdt
    k3_dRdt, k3_d2Rdt2, k3_dTsdt = rayleigh_plesset_derivatives(R3, dRdt3, T_s3)
    
    # k4
    R4 = R + dt * k3_dRdt
    dRdt4 = dRdt + dt * k3_d2Rdt2
    T_s4 = T_s + dt * k3_dTsdt
    k4_dRdt, k4_d2Rdt2, k4_dTsdt = rayleigh_plesset_derivatives(R4, dRdt4, T_s4)
    
    # Update using weighted average
    R_new = R + (dt/6.0) * (k1_dRdt + 2*k2_dRdt + 2*k3_dRdt + k4_dRdt)
    dRdt_new = dRdt + (dt/6.0) * (k1_d2Rdt2 + 2*k2_d2Rdt2 + 2*k3_d2Rdt2 + k4_d2Rdt2)
    T_s_new = T_s + (dt/6.0) * (k1_dTsdt + 2*k2_dTsdt + 2*k3_dTsdt + k4_dTsdt)
    
    # Apply physical bounds
    R_new = max(R_new, 1e-12)
    T_s_new = max(T_s_new, T_b)
    
    return R_new, dRdt_new, T_s_new

def print_state(t, R, dRdt, T_s):
    """Print current state of the simulation"""
    print(f"t = {t:.6e} s, R = {R:.6e} m, dR/dt = {dRdt:.6e} m/s, T_s = {T_s:.2f} K")

def solve_bubble_dynamics():
    """Main simulation function using adaptive time stepping"""
    print("Solving bubble dynamics with RK4 integration...")
    
    # Time span - use logarithmic time steps
    t_start = 1e-9
    t_end = 1.0
    n_points = 500
    t_eval = np.logspace(np.log10(t_start), np.log10(t_end), n_points)
    
    # Initialize arrays
    R = np.zeros(n_points)
    dRdt = np.zeros(n_points)
    T_s = np.zeros(n_points)
    
    # Set initial conditions
    R[0] = R_initial
    dRdt[0] = dRdt_initial
    T_s[0] = T_inf
    
    # Integrate with progress bar
    print_state(t_eval[0], R[0], dRdt[0], T_s[0])
    
    with tqdm(total=n_points-1, desc="Integrating") as pbar:
        for i in range(1, n_points):
            # Get time interval
            t_current = t_eval[i-1]
            t_next = t_eval[i]
            dt = t_next - t_current
            
            # Adaptive substeps for stability
            substeps = max(1, min(100, int(dt / (1e-10 * R[i-1]))))
            dt_sub = dt / substeps
            
            # Current state
            R_current = R[i-1]
            dRdt_current = dRdt[i-1]
            T_s_current = T_s[i-1]
            
            # Multiple smaller RK4 steps
            for _ in range(substeps):
                R_current, dRdt_current, T_s_current = rk4_step(
                    R_current, dRdt_current, T_s_current, dt_sub
                )
            
            # Store results
            R[i] = R_current
            dRdt[i] = dRdt_current
            T_s[i] = T_s_current
            
            # Update progress
            if i % 50 == 0:
                print_state(t_next, R_current, dRdt_current, T_s_current)
            
            pbar.update(1)
    
    # Print final state
    print_state(t_eval[-1], R[-1], dRdt[-1], T_s[-1])
    
    return t_eval, R, dRdt, T_s

def plot_results(t, R, dRdt, T_s):
    """Create plots of the simulation results"""
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot 1: Bubble Radius vs Time
    axs[0].loglog(t, R * 100, 'b-', linewidth=2)  # Convert to cm
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Bubble Radius (cm)')
    axs[0].set_title('Bubble Radius vs Time')
    axs[0].grid(True, which="both", ls="--")
    
    # Plot 2: Bubble Wall Velocity vs Time
    axs[1].loglog(t, np.abs(dRdt) * 100, 'r-', linewidth=2)  # Convert to cm/s
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Wall Velocity (cm/s)')
    axs[1].set_title('Bubble Wall Velocity vs Time')
    axs[1].grid(True, which="both", ls="--")
    
    # Plot 3: Surface Temperature vs Time
    axs[2].semilogx(t, T_s, 'g-', linewidth=2)
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Surface Temperature (K)')
    axs[2].set_title('Surface Temperature vs Time')
    axs[2].grid(True, which="both", ls="--")
    axs[2].axhline(y=T_b, color='k', linestyle='--', label=f'T_boil = {T_b} K')
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig('bubble_dynamics_results.png', dpi=300)
    print("Plot saved as 'bubble_dynamics_results.png'")
    plt.show()

def main():
    """Main function"""
    print("======== Bubble Dynamics Simulation ========")
    print(f"Liquid sodium at {p_inf/1e5:.1f} bar")
    print(f"Initial temperature: {T_inf} K (superheat: {T_inf - T_b} K)")
    print(f"Initial bubble radius: {R_initial*1e6:.1f} μm")
    
    # Solve the system
    t, R, dRdt, T_s = solve_bubble_dynamics()
    
    # Calculate key values
    max_R = np.max(R) * 1000  # mm
    max_v = np.max(np.abs(dRdt)) * 100  # cm/s
    min_T = np.min(T_s)
    
    # Print summary
    print("\n======== Results Summary ========")
    print(f"Maximum bubble radius: {max_R:.3f} mm")
    print(f"Maximum bubble wall velocity: {max_v:.1f} cm/s")
    print(f"Minimum surface temperature: {min_T:.1f} K")
    
    # Plot results
    plot_results(t, R, dRdt, T_s)

if __name__ == "__main__":
    main()
