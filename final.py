import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from tqdm import tqdm  # Import tqdm for progress bar

# Define fluid properties and simulation cases
fluids = {
    'water': {
        'cases': [
            {'name': 'Water 20K', 'p_inf': 1.013e5, 'Delta_T': 20, 'color': 'blue'},
            {'name': 'Water 50K', 'p_inf': 1.013e5, 'Delta_T': 50, 'color': 'orange'},
            {'name': 'Water 100K', 'p_inf': 1.013e5, 'Delta_T': 100, 'color': 'green'}
        ],
        'properties': {
            'rho_l': 958.4,        # liquid density (kg/m³)
            'c_l': 4216,           # specific heat capacity of liquid (J/kg/K)
            'lambda_l': 0.68,      # thermal conductivity of liquid (W/m/K)
            'L': 2.257e6,          # latent heat of vaporization (J/kg) - will use the interpolation function
            'R_gas': 461.5,        # specific gas constant for vapor (J/kg/K)
            'sigma': 0.0589,       # surface tension (N/m)
            'eta_l': 0,            # dynamic viscosity (Pa.s)
            'T_sat': 373.15,       # saturation temperature at 1 atm (K)
            'beta': 1000           # thermal enhancement factor (assumed)
        },
        'initial_radii': {
            20: 3e-6,    # 3 micrometers for ΔT = 20K
            50: 3e-7,    # 300 nanometers for ΔT = 50K
            100: 8e-8    # 80 nanometers for ΔT = 100K
        }
    },
    'sodium': {
        'cases': [
            {
                'name': 'Na 0.5atm 340K', 
                'p_inf': 0.5*1.013e5, 
                'Delta_T': 340, 
                'color': 'red',
                # Pressure-specific properties for 0.5 atm
                'properties': {
                    'rho_l': 756,          # liquid sodium density (kg/m³) at 0.5 atm
                    'c_l': 1261,           # specific heat capacity (J/kg/K) at 0.5 atm
                    'lambda_l': 60,        # thermal conductivity (W/m/K) at 0.5 atm
                    'L': 4e6,              # latent heat (J/kg) at 0.5 atm - will use interpolation
                    'R_gas': 361.5,        # gas constant of sodium vapor (J/kg/K)
                    'sigma': 0.16,         # surface tension (N/m) at 0.5 atm
                    'eta_l': 0,            # dynamic viscosity (Pa.s)
                    'T_sat': 1107,         # saturation temperature (K) at 0.5 atm
                    'beta': 1000           # thermal enhancement factor (assumed)
                }
            },
            {
                'name': 'Na 2atm 90K', 
                'p_inf': 2*1.013e5, 
                'Delta_T': 90, 
                'color': 'purple',
                # Pressure-specific properties for 2 atm
                'properties': {
                    'rho_l': 719,          # liquid sodium density (kg/m³) at 2 atm
                    'c_l': 1292,           # specific heat capacity (J/kg/K) at 2 atm
                    'lambda_l': 62,        # thermal conductivity (W/m/K) at 2 atm
                    'L': 4.1e6,            # latent heat (J/kg) at 2 atm - will use interpolation
                    'R_gas': 361.5,        # gas constant of sodium vapor (J/kg/K)
                    'sigma': 0.158,        # surface tension (N/m) at 2 atm
                    'eta_l': 0,            # dynamic viscosity (Pa.s)
                    'T_sat': 1253,         # saturation temperature (K) at 2 atm
                    'beta': 1000           # thermal enhancement factor (assumed)
                }
            },
            {
                'name': 'Na 4.5atm 15K', 
                'p_inf': 4.5*1.013e5, 
                'Delta_T': 15, 
                'color': 'brown',
                # Pressure-specific properties for 4.5 atm
                'properties': {
                    'rho_l': 693,          # liquid sodium density (kg/m³) at 4.5 atm
                    'c_l': 1340,           # specific heat capacity (J/kg/K) at 4.5 atm
                    'lambda_l': 64,        # thermal conductivity (W/m/K) at 4.5 atm
                    'L': 4.2e6,            # latent heat (J/kg) at 4.5 atm - will use interpolation
                    'R_gas': 361.5,        # gas constant of sodium vapor (J/kg/K)
                    'sigma': 0.155,        # surface tension (N/m) at 4.5 atm
                    'eta_l': 0,            # dynamic viscosity (Pa.s)
                    'T_sat': 1352,         # saturation temperature (K) at 4.5 atm
                    'beta': 1000           # thermal enhancement factor (assumed)
                }
            }
        ],
        'initial_radii': {
            340: 4.1e-7,
            90: 2.87e-6,
            15: 1.56e-5
        }
    }
}

# Water latent heat interpolation data
water_temperature_K = np.array([
    273.15, 275, 280, 285, 290, 295, 300, 305, 310, 315, 320,
    325, 330, 335, 340, 345, 350, 355, 360, 365, 370, 373.15,
    375, 380, 385, 390, 400, 410, 420, 430, 440, 450, 460, 470,
    480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590,
    600, 610, 620, 625, 630, 635, 640, 645, 647.3
])

# Latent heat in kJ/kg
water_latent_heat_kJ_per_kg = np.array([
    2502, 2497, 2485, 2473, 2461, 2449, 2438, 2426, 2414, 2402, 2390,
    2378, 2366, 2354, 2342, 2329, 2317, 2304, 2291, 2278, 2265, 2257,
    2252, 2239, 2225, 2212, 2183, 2153, 2123, 2091, 2059, 2024, 1989,
    1951, 1912, 1870, 1825, 1779, 1730, 1679, 1622, 1564, 1499, 1429,
    1353, 1274, 1176, 1068, 941, 858, 781, 683, 560, 361, 0
])

# Create interpolation function for water
water_latent_heat_interp = interp1d(
    water_temperature_K, water_latent_heat_kJ_per_kg * 1000,  # Convert to J/kg
    kind='linear', bounds_error=False, fill_value=(water_latent_heat_kJ_per_kg[0]*1000, 0)
)

# This is a placeholder - you'll need to add actual sodium data
sodium_temperature_K = np.array([800, 900, 1000, 1100, 1156, 1200, 1300, 1400])
sodium_latent_heat_kJ_per_kg = np.array([4500, 4400, 4300, 4200, 4100, 4000, 3800, 3600])

# Create interpolation function for sodium
sodium_latent_heat_interp = interp1d(
    sodium_temperature_K, sodium_latent_heat_kJ_per_kg * 1000,  # Convert to J/kg
    kind='linear', bounds_error=False, fill_value=(sodium_latent_heat_kJ_per_kg[0]*1000, sodium_latent_heat_kJ_per_kg[-1]*1000)
)

# Function to get latent heat based on fluid type and temperature
def get_latent_heat(fluid_type, temperature_K):
    """
    Get latent heat for a given fluid at a specific temperature.
    
    Args:
        fluid_type (str): 'water' or 'sodium'
        temperature_K (float): Temperature in Kelvin
        
    Returns:
        float: Latent heat in J/kg
    """
    if fluid_type == 'water':
        return water_latent_heat_interp(temperature_K)
    elif fluid_type == 'sodium':
        return sodium_latent_heat_interp(temperature_K)
    else:
        raise ValueError(f"Unknown fluid type: {fluid_type}")

# Define the system of ODEs
def bubble_odes(t, y, fluid_type, properties, T_inf):
    R, R_dot, rho_v, T_s = y
    
    # Extract fluid properties
    rho_l = properties['rho_l']
    c_l = properties['c_l']
    lambda_l = properties['lambda_l']
    R_gas = properties['R_gas']
    sigma = properties['sigma']
    eta_l = properties['eta_l']
    p_inf = properties['p_inf']
    beta = properties['beta']

    if R <= 0:
        return [0, 0, 0, 0]

    # Temperature gradient at interface (assuming quasi-steady conduction)
    dTdr = (T_inf - T_s) / R
     
    # Get latent heat at interface temperature
    L = get_latent_heat(fluid_type, T_s)
    
    # Mass flux at the interface
    j = lambda_l * dTdr / L

    # Liquid velocity at interface
    v_lR = R_dot - j / rho_l

    # Vapor pressure from ideal gas law
    p_v = rho_v * R_gas * T_s

    # Interface pressure balance (includes surface tension and viscosity terms)
    p_lR = p_v + j * v_lR - 2 * sigma / R - 4 * eta_l * v_lR / R

    # Rayleigh-Plesset type equation (modified)
    dR_dt = R_dot
    R_ddot = (p_lR - p_inf) / (rho_l * R) + (v_lR**2) / (2 * R) - 2 * v_lR * R_dot / R

    # Mass conservation inside bubble
    drho_v_dt = (3 * (j - rho_v * v_lR)) / R

    # Energy conservation at the interface (liquid side energy loss)
    alpha = lambda_l / (rho_l * c_l)  # thermal diffusivity of liquid
    Grad_dist = beta * np.sqrt(np.pi * alpha * t)  # characteristic length scale for diffusion
    dT_s_dt = (lambda_l / (rho_l * c_l)) * ((T_inf - T_s) / (Grad_dist**2)) - ((v_lR) * ((T_inf - T_s)/Grad_dist))

    return [dR_dt, R_ddot, drho_v_dt, dT_s_dt]

# Calculate mass flux for given conditions (matches bubble_odes implementation)
def calculate_mass_flux(t, R, T_s, fluid_type, properties, T_inf):
    """
    Calculate the mass flux at the bubble interface using the same formula as in bubble_odes
    
    Args:
        t (float): Time
        R (float): Bubble radius
        T_s (float): Interface temperature
        fluid_type (str): 'water' or 'sodium'
        properties (dict): Fluid properties
        T_inf (float): Bulk liquid temperature
        
    Returns:
        float: Mass flux (kg/m²/s)
    """
    # Extract needed properties
    lambda_l = properties['lambda_l']
    
    # Temperature gradient at interface (same as in bubble_odes)
    dTdr = (T_inf - T_s) / R
    
    # Get latent heat at interface temperature
    L = get_latent_heat(fluid_type, T_s)
    
    # Mass flux at the interface (same formula as in bubble_odes)
    j = lambda_l * dTdr / L
    
    return j

# Time span for simulation
t_span = (1e-9, 1e-4)
points = 5000
t_eval = np.logspace(np.log10(t_span[0]), np.log10(t_span[1]), points)

# Function to run simulation and generate plots
def run_simulation_and_plot(selected_fluids=['water']):
    # Create figures for each plot type
    radius_fig = plt.figure(figsize=(10, 8))
    velocity_fig = plt.figure(figsize=(10, 8))
    mass_flux_fig = plt.figure(figsize=(10, 8))
    
    # Create separate non-dimensional figures for each fluid
    non_dim_radius_figs = {}
    non_dim_velocity_figs = {}
    for fluid_type in selected_fluids:
        non_dim_radius_figs[fluid_type] = plt.figure(figsize=(10, 8))
        non_dim_velocity_figs[fluid_type] = plt.figure(figsize=(10, 8))
    
    # Keep track of all cases to include in legends
    all_cases = []
    
    # Define non-dimensional parameters for each case
    # Format: [fluid_type, Delta_T, p_inf_bar, mu, alpha]
    non_dim_params = {
        'water': {
            20: {'mu': 5.324, 'alpha': 7e8},
            50: {'mu': 4.756, 'alpha': 8e8},
            100: {'mu': 5.491, 'alpha': 9e8}
        },
        'sodium': {
            340: {'mu': 0.000403, 'alpha': 1e8},
            90: {'mu': 0.00463, 'alpha': 1.03e7},
            15: {'mu': 0.0501, 'alpha': 3e6}
        }
    } #Taken from Plesset's paper
    
    # Count total cases for progress tracking
    total_cases = sum(len(fluids[fluid]['cases']) for fluid in selected_fluids)
    case_count = 0
    
    # Process each selected fluid
    for fluid_type in selected_fluids:
        fluid = fluids[fluid_type]
        cases = fluid['cases']
        initial_radii = fluid['initial_radii']
        
        # Process each case for the current fluid
        for case in cases:
            case_count += 1
            # Set up parameters for this case
            p_inf = case['p_inf']
            Delta_T = case['Delta_T']
            name = case['name']
            
            # Get properties based on fluid type
            if fluid_type == 'water':
                # For water, use the global properties
                properties = fluid['properties'].copy()
                properties['p_inf'] = p_inf  # Update pressure
                T_sat = properties['T_sat']
            else:  # for sodium
                # For sodium, each case has its own properties
                properties = case['properties'].copy()
                properties['p_inf'] = p_inf  # Ensure pressure is set
                T_sat = properties['T_sat']
            
            T_inf = T_sat + Delta_T
            color = case['color']
            
            # Add this case to our tracking list
            all_cases.append(case)
            
            # Initial conditions
            R0 = initial_radii[Delta_T]
            R_dot0 = 0.0
            T_s0 = T_sat
            rho_v0 = p_inf / (properties['R_gas'] * T_sat)  # initial vapor density
            
            # Initial conditions vector: [R, R_dot, rho_v, T_s]
            y0 = [R0, R_dot0, rho_v0, T_s0]
            
            # Show detailed progress
            print(f"[{case_count}/{total_cases}] Solving for {name} (ΔT={Delta_T}K, P={p_inf/1.013e5:.2f} atm)")
            
            # Solve the ODE system with tqdm to show progress during integration
            with tqdm(total=100, desc=f"Integration progress", bar_format='{l_bar}{bar:30}{r_bar}') as pbar:
                def progress_callback(t, y):
                    # Update progress bar based on current time in the simulation
                    progress = min(100, int(100 * (np.log10(t) - np.log10(t_span[0])) / 
                                          (np.log10(t_span[1]) - np.log10(t_span[0]))))
                    pbar.update(progress - pbar.n)
                    return False  # Continue integration
                
                sol = solve_ivp(
                    lambda t, y: bubble_odes(t, y, fluid_type, properties, T_inf), 
                    t_span, y0, t_eval=t_eval, method='Radau', rtol=1e-6, atol=1e-9,
                    events=progress_callback
                )
            
            # Extract results
            times = sol.t
            radii = sol.y[0]
            velocities = sol.y[1]
            temperatures = sol.y[3]
            
            # Calculate mass flux for each time step
            print(f"Calculating mass flux values for {name}...")
            mass_fluxes = [calculate_mass_flux(t, R, T_s, fluid_type, properties, T_inf) 
                           for t, R, T_s in zip(times, radii, temperatures)]
            
            # Plot radius vs time
            plt.figure(radius_fig.number)
            plt.plot(times, radii, label=name, color=color)
            
            # Plot velocity vs time
            plt.figure(velocity_fig.number)
            plt.plot(times, velocities, label=name, color=color)
            
            # Plot mass flux vs time
            plt.figure(mass_flux_fig.number)
            plt.plot(times, mass_fluxes, label=name, color=color)
            
            # Get non-dimensional parameters for this case
            mu = non_dim_params[fluid_type][Delta_T]['mu']
            alpha = non_dim_params[fluid_type][Delta_T]['alpha']
            
            # Calculate non-dimensional results
            non_dim_radii = (mu**2) * radii / R0
            non_dim_times = alpha * (mu**2) * times
            non_dim_velocities = velocities / (alpha * R0)
            
            # Plot non-dimensional radius vs non-dimensional time on fluid-specific figure
            plt.figure(non_dim_radius_figs[fluid_type].number)
            plt.plot(non_dim_times, non_dim_radii, label=name, color=color)
            
            # Plot non-dimensional velocity vs non-dimensional time on fluid-specific figure
            plt.figure(non_dim_velocity_figs[fluid_type].number)
            plt.plot(non_dim_times, non_dim_velocities, label=name, color=color)
            
            print(f"Completed {name} ({case_count}/{total_cases})\n")
    
    # Finalize radius plot
    plt.figure(radius_fig.number)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('Bubble radius (m)')
    plt.title('Bubble Growth in Superheated Liquid (with Mass & Energy Transfer)')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig('bubble_radius_vs_time.png', dpi=300)
    
    # Finalize velocity plot
    plt.figure(velocity_fig.number)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('Bubble wall velocity dR/dt (m/s)')
    plt.title('Bubble Wall Velocity in Superheated Liquid (with Mass & Energy Transfer)')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig('bubble_velocity_vs_time.png', dpi=300)
    
    # Finalize mass flux plot
    plt.figure(mass_flux_fig.number)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('Mass flux j (kg/m²/s)')
    plt.title('Interface Mass Flux in Superheated Liquid (with Mass & Energy Transfer)')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig('mass_flux_vs_time.png', dpi=300)
    
    # Finalize fluid-specific non-dimensional plots
    for fluid_type in selected_fluids:
        # Radius plots
        plt.figure(non_dim_radius_figs[fluid_type].number)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Non-dimensional time $\\tilde{t} = \\alpha \\mu^2 t$')
        plt.ylabel('Non-dimensional radius $\\tilde{R} = \\mu^2 R/R_0$')
        plt.title(f'Non-dimensional Bubble Radius Growth - {fluid_type.capitalize()}')
        plt.grid(True, which='both', ls='--')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'non_dim_radius_{fluid_type}.png', dpi=300)
        
        # Velocity plots
        plt.figure(non_dim_velocity_figs[fluid_type].number)
        plt.xscale('log')
        plt.ylim(0, 4.0)  # Based on expected maximum non-dimensional velocity
        plt.xlabel('Non-dimensional time $\\tilde{t} = \\alpha \\mu^2 t$')
        plt.ylabel('Non-dimensional velocity $\\frac{d\\tilde{R}}{d\\tilde{t}}$')
        plt.title(f'Non-dimensional Bubble Wall Velocity - {fluid_type.capitalize()}')
        plt.grid(True, which='both', ls='--')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'non_dim_velocity_{fluid_type}.png', dpi=300)
    
    print("All simulations and plots completed successfully!")
    return radius_fig, velocity_fig, mass_flux_fig, non_dim_radius_figs, non_dim_velocity_figs

# Time span for simulation
t_span = (1e-9, 1e-4)
points = 5000
t_eval = np.logspace(np.log10(t_span[0]), np.log10(t_span[1]), points)

# Run the simulation with both fluids
if __name__ == "__main__":
    # Run simulation for water only (original case)
    # run_simulation_and_plot(['water'])
    
    # Run simulation for both water and sodium
    run_simulation_and_plot(['water', 'sodium'])
    
    # Display all plots
    plt.show()
