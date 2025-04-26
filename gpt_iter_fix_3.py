#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from tqdm import tqdm
from joblib import Parallel, delayed
import concurrent.futures

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

def vapor_pressure(T):
    return 10**(A - B/(T + C)) * 1e5  # Convert bar to Pa

def dSdt(t, S):
    """Define the system of ODEs."""
    R = max(S[0], 1e-10)  # Prevent division by zero
    dRdt = S[1]
    T_s = S[2]
    p_v = vapor_pressure(T_s)
    d2Rdt2 = (1/rho) * (p_v - p_inf - 2*sigma/R) / R - 1.5 * (dRdt**2) / R
    dTsdt = -L * rho * dRdt / k * (T_s - T_b) / R
    return [dRdt, d2Rdt2, dTsdt]

class ProgressIVP:
    def __init__(self, fun, t_span, y0, t_eval=None, **kwargs):
        self.fun = fun
        self.t_span = t_span
        self.y0 = y0
        self.t_eval = t_eval
        self.kwargs = kwargs
        self.n_steps = 0
        self.t_steps = []
        self.y_steps = []
        
    def wrapper(self, t, y):
        """Wrapper function that counts calls to the ODE function."""
        self.t_steps.append(t)
        self.n_steps += 1
        progress_bar.set_description(f"t = {t:.2e} s | steps = {self.n_steps}")
        progress_bar.update(1)
        return self.fun(t, y)
    
    def solve(self, max_steps=10000):
        """Solve the IVP with progress bar."""
        self.n_steps = 0
        self.t_steps = []
        self.y_steps = []
        global progress_bar
        progress_bar = tqdm(total=max_steps, desc="Solving ODEs", dynamic_ncols=True)
        
        result = solve_ivp(
            self.wrapper, 
            self.t_span, 
            self.y0,
            t_eval=self.t_eval,
            **self.kwargs
        )
        
        progress_bar.close()
        return result

def solve_stage(t_start, t_end, S0, rtol, atol, t_points):
    """Solve a single stage with custom tolerances and points."""
    t_span = [t_start, t_end]
    t_eval = np.logspace(np.log10(t_start), np.log10(t_end), t_points)
    
    solver = ProgressIVP(
        dSdt,
        t_span,
        S0,
        t_eval=t_eval,
        method='RK45',
        rtol=rtol,
        atol=atol
    )
    
    solution = solver.solve(max_steps=3000)
    return solution

def solve_bubble_dynamics_two_stage():
    """Solve the bubble dynamics equations using two-stage fast approach."""
    # Stage 1: Early time (critical phase)
    S0 = [R_initial, dRdt_initial, T_inf]
    sol1 = solve_stage(1e-9, 1e-5, S0, 1e-6, 1e-9, 300)
    
    # Stage 2: Later time (bubble already formed)
    S0_stage2 = [sol1.y[0][-1], sol1.y[1][-1], sol1.y[2][-1]]
    sol2 = solve_stage(1e-5, 1e0, S0_stage2, 1e-5, 1e-8, 300)
    
    # Combine results
    t_combined = np.concatenate((sol1.t, sol2.t))
    y_combined = np.hstack((sol1.y, sol2.y))
    
    class CombinedSolution:
        t = t_combined
        y = y_combined
        success = sol1.success and sol2.success
        
    return CombinedSolution()

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

def parallel_solve(*args):
    """Parallelize the solving process."""
    return solve_bubble_dynamics_two_stage()

def main():
    """Main function to run the two-stage fast simulation."""
    print("Solving Rayleigh-Plesset equation for vapor bubble in liquid sodium (TWO-STAGE FAST MODE)...")
    
    # Parallelize the simulation process (if needed, we can divide work into multiple chunks)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future = executor.submit(parallel_solve)
        result = future.result()

    if result.success:
        print("Simulation completed successfully!")
        plot_results(result)
        
        max_radius = np.max(result.y[0]) * 1000  # mm
        max_velocity = np.max(result.y[1]) * 100  # cm/s
        min_temperature = np.min(result.y[2])
        
        print(f"Maximum bubble radius: {max_radius:.3f} mm")
        print(f"Maximum growth rate: {max_velocity:.3f} cm/s")
        print(f"Minimum surface temperature: {min_temperature:.1f} K")
    else:
        print("Simulation failed!")

if __name__ == "__main__":
    main()
