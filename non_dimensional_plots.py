import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from tqdm import tqdm

# Import from final.py but don't run the simulations
from final import fluids, get_latent_heat, bubble_odes

# Define the non-dimensional parameters for each case from the provided tables
# Format: [fluid, pressure (bar), superheat (K), mu, alpha]
cases = [
    ["Water", 1.0, 20, 5.324, 8.42e11],  # Changed from 30K to 20K to match final.py keys
    ["Water", 1.0, 50, 4.756, 2.08e12],
    ["Water", 1.0, 100, 5.491, 1.42e13],
    ["Sodium", 0.5, 340, 0.000403, 1.20e6],
    ["Sodium", 2.0, 90, 0.00463, 1.03e7],
    ["Sodium", 4.5, 15, 0.0501, 3.49e7]
]

# Define time span for simulation
t_span = (1e-9, 1)
points = 5000
t_eval = np.logspace(np.log10(t_span[0]), np.log10(t_span[1]), points)

# Run simulations and store dimensional and non-dimensional results
dimensional_results = []
non_dimensional_results = []

for case_info in tqdm(cases, desc="Processing cases"):
    fluid_type, p_inf_bar, Delta_T, mu, alpha = case_info
    
    # Convert bar to Pa
    p_inf = p_inf_bar * 1.013e5
    
    # Get properties based on fluid type
    if fluid_type.lower() == "water":
        fluid_data = fluids['water']
        properties = fluid_data['properties'].copy()
        properties['p_inf'] = p_inf
        T_sat = properties['T_sat']
        initial_radii = fluid_data['initial_radii']
        R0 = initial_radii[int(Delta_T)]  # Initial radius (dimensional)
    else:  # sodium
        fluid_data = fluids['sodium']
        for case in fluid_data['cases']:
            if abs(case['p_inf'] - p_inf) < 0.01*p_inf and abs(case['Delta_T'] - Delta_T) < 0.01*Delta_T:
                properties = case['properties'].copy()
                properties['p_inf'] = p_inf
                T_sat = properties['T_sat']
                initial_radii = fluid_data['initial_radii']
                R0 = initial_radii[int(Delta_T)]
                break
    
    # Calculate T_inf
    T_inf = T_sat + Delta_T
    
    # Initial conditions
    R_dot0 = 0.0
    T_s0 = T_sat
    rho_v0 = p_inf / (properties['R_gas'] * T_sat)  # initial vapor density
    
    # Initial conditions vector: [R, R_dot, rho_v, T_s]
    y0 = [R0, R_dot0, rho_v0, T_s0]
    
    # Solve the ODE system
    sol = solve_ivp(
        lambda t, y: bubble_odes(t, y, fluid_type.lower(), properties, T_inf), 
        t_span, y0, t_eval=t_eval, method='Radau', rtol=1e-6, atol=1e-9
    )
    
    # Extract dimensional results
    times = sol.t
    radii = sol.y[0]
    velocities = sol.y[1]
    
    # Calculate non-dimensional results using the provided equations
    # R̃ = μ²R/R₀,  t̃ = αμ²t
    non_dim_radii = (mu**2) * radii / R0
    non_dim_times = alpha * (mu**2) * times
    
    # Calculate non-dimensional velocity using the derivative of R̃ with respect to t̃
    # dR̃/dt̃ = (dR̃/dt) / (dt̃/dt) = (μ²/R₀) * (dR/dt) / (αμ²) = (1/(αR₀)) * dR/dt
    non_dim_velocities = velocities / (alpha * R0)
    
    # Store results
    case_name = f"{fluid_type} {Delta_T}K {p_inf_bar}bar"
    dimensional_results.append({
        'name': case_name,
        'times': times,
        'radii': radii,
        'velocities': velocities,
    })
    
    non_dimensional_results.append({
        'name': case_name,
        'non_dim_times': non_dim_times,
        'non_dim_radii': non_dim_radii,
        'non_dim_velocities': non_dim_velocities,
    })

# Create color map for better visualization
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Plot non-dimensional radius vs non-dimensional time
plt.figure(figsize=(12, 8))
for i, result in enumerate(non_dimensional_results):
    plt.plot(result['non_dim_times'], result['non_dim_radii'], 
             label=result['name'], color=colors[i], linewidth=2)
    
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Non-dimensional time $\\tilde{t} = \\alpha \\mu^2 t$', fontsize=14)
plt.ylabel('Non-dimensional radius $\\tilde{R} = \\mu^2 R/R_0$', fontsize=14)
plt.title('Non-dimensional Bubble Radius Growth', fontsize=16)
plt.grid(True, which='both', ls='--')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('non_dimensional_radius_growth.png', dpi=300)

# Plot non-dimensional velocity vs non-dimensional time
plt.figure(figsize=(12, 8))
for i, result in enumerate(non_dimensional_results):
    plt.plot(result['non_dim_times'], result['non_dim_velocities'], 
             label=result['name'], color=colors[i], linewidth=2)
    
plt.xscale('log')
plt.ylim(0, 1.0)  # Based on the reference image showing maximum value around 0.8
plt.xlabel('Non-dimensional time $\\tilde{t} = \\alpha \\mu^2 t$', fontsize=14)
plt.ylabel('Non-dimensional velocity $\\frac{d\\tilde{R}}{d\\tilde{t}}$', fontsize=14)
plt.title('Non-dimensional Bubble Wall Velocity', fontsize=16)
plt.grid(True, which='both', ls='--')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('non_dimensional_velocity.png', dpi=300)

# Plot comparison of all dimensional and non-dimensional results in a 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# Dimensional radius
ax = axs[0, 0]
for i, result in enumerate(dimensional_results):
    ax.plot(result['times'], result['radii'], 
            label=result['name'], color=colors[i], linewidth=2)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Bubble radius (m)', fontsize=12)
ax.set_title('Dimensional Bubble Radius', fontsize=14)
ax.grid(True, which='both', ls='--')
ax.legend(fontsize=10)

# Non-dimensional radius
ax = axs[0, 1]
for i, result in enumerate(non_dimensional_results):
    ax.plot(result['non_dim_times'], result['non_dim_radii'], 
            label=result['name'], color=colors[i], linewidth=2)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Non-dimensional time $\\tilde{t}$', fontsize=12)
ax.set_ylabel('Non-dimensional radius $\\tilde{R}$', fontsize=12)
ax.set_title('Non-dimensional Bubble Radius', fontsize=14)
ax.grid(True, which='both', ls='--')
ax.legend(fontsize=10)

# Dimensional velocity
ax = axs[1, 0]
for i, result in enumerate(dimensional_results):
    ax.plot(result['times'], result['velocities'], 
            label=result['name'], color=colors[i], linewidth=2)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Bubble wall velocity (m/s)', fontsize=12)
ax.set_title('Dimensional Bubble Wall Velocity', fontsize=14)
ax.grid(True, which='both', ls='--')
ax.legend(fontsize=10)

# Non-dimensional velocity
ax = axs[1, 1]
for i, result in enumerate(non_dimensional_results):
    ax.plot(result['non_dim_times'], result['non_dim_velocities'], 
            label=result['name'], color=colors[i], linewidth=2)
ax.set_xscale('log')
ax.set_ylim(0, 1.0)
ax.set_xlabel('Non-dimensional time $\\tilde{t}$', fontsize=12)
ax.set_ylabel('Non-dimensional velocity $\\frac{d\\tilde{R}}{d\\tilde{t}}$', fontsize=12)
ax.set_title('Non-dimensional Bubble Wall Velocity', fontsize=14)
ax.grid(True, which='both', ls='--')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('dimensional_vs_non_dimensional_comparison.png', dpi=300)

plt.show()