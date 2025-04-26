#!/usr/bin/env python3
"""
Fast GPU-accelerated Rayleigh-Plesset simulation
Using Numba's CUDA to parallelize ODE right-hand side evaluations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm
from numba import cuda, float64

# Check for GPU
if not cuda.is_available():
    raise RuntimeError("CUDA-capable GPU not available!")

# Constants
rho = 800  # kg/m³
sigma = 0.2  # N/m
k = 60  # W/m·K
D = 6.5e-7  # m²/s
L = 4.3e6  # J/kg
p_inf = 0.5e5  # Pa
T_inf = 1396  # K
T_b = 1056  # K
R_initial = 1e-6  # m
dRdt_initial = 0  # m/s

A = 4.52
B = 5100
C = 0

# CUDA kernel for vapor pressure
@cuda.jit(device=True)
def vapor_pressure(T):
    return 10**(A - B/(T + C)) * 1e5

# CUDA kernel for dSdt
@cuda.jit
def gpu_dSdt_kernel(t, S, dSdt_out):
    idx = cuda.grid(1)
    if idx >= S.shape[1]:
        return
    
    R = max(S[0, idx], 1e-10)
    dRdt = S[1, idx]
    T_s = S[2, idx]
    
    p_v = vapor_pressure(T_s)
    d2Rdt2 = (1.0/rho) * (p_v - p_inf - 2.0*sigma/R)/R - 1.5 * (dRdt**2) / R
    dTsdt = -L * rho * dRdt / k * (T_s - T_b) / R
    
    dSdt_out[0, idx] = dRdt
    dSdt_out[1, idx] = d2Rdt2
    dSdt_out[2, idx] = dTsdt

# CPU-side function to call GPU kernel
def gpu_dSdt(t, S_flat):
    S = S_flat.reshape(3, -1)
    dSdt_out = np.zeros_like(S)
    
    threads_per_block = 128
    blocks_per_grid = (S.shape[1] + (threads_per_block - 1)) // threads_per_block
    
    S_device = cuda.to_device(S)
    dSdt_device = cuda.to_device(dSdt_out)
    
    gpu_dSdt_kernel[blocks_per_grid, threads_per_block](t, S_device, dSdt_device)
    
    dSdt_out = dSdt_device.copy_to_host()
    return dSdt_out.reshape(-1)

# CPU-side wrapper to solve a single bubble
def solve_bubble_gpu():
    t_span = (1e-9, 1e-0)
    S0 = np.array([R_initial, dRdt_initial, T_inf])
    
    def wrapper(t, S):
        return gpu_dSdt(t, S)
    
    t_eval = np.logspace(np.log10(t_span[0]), np.log10(t_span[1]), 500)
    
    print("Solving with GPU acceleration...")
    
    sol = solve_ivp(
        wrapper,
        t_span,
        S0,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9
    )
    
    return sol

# Plotting
def plot_solution(sol):
    t = sol.t
    R = sol.y[0]
    dRdt = sol.y[1]
    T_s = sol.y[2]
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 14))
    
    axs[0].loglog(t, R*100, 'b-')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Bubble Radius (cm)')
    axs[0].set_title('Bubble Radius vs Time (GPU)')
    axs[0].grid(True, which="both", ls="--")
    
    axs[1].loglog(t, abs(dRdt)*100, 'r-')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('|dR/dt| (cm/s)')
    axs[1].set_title('Growth Rate vs Time (GPU)')
    axs[1].grid(True, which="both", ls="--")
    
    axs[2].semilogx(t, T_s, 'g-')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Surface Temperature (K)')
    axs[2].set_title('Surface Temperature vs Time (GPU)')
    axs[2].grid(True, which="both", ls="--")
    axs[2].axhline(y=T_b, color='k', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('bubble_dynamics_gpu.png', dpi=300)
    plt.show()

# Main
def main():
    sol = solve_bubble_gpu()
    
    if sol.success:
        print("Simulation completed successfully (GPU)!")
        plot_solution(sol)
    else:
        print("Simulation failed.")

if __name__ == "__main__":
    main()
