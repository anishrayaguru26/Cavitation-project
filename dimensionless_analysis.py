import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math

# Import the interpolation functions and fluid properties from final.py but don't run the simulations
from final import fluids, water_temperature_K, water_latent_heat_kJ_per_kg, sodium_temperature_K, sodium_latent_heat_kJ_per_kg, get_latent_heat

# Define thermal diffusivity (D) values
# D = lambda_l / (rho_l * c_l) = thermal conductivity / (density * specific heat)
# For sodium, it's given as 1.1e-5 m²/s
D_sodium = 1.1e-5  # m²/s

# For water, calculate it based on properties
water_props = fluids['water']['properties']
D_water = water_props['lambda_l'] / (water_props['rho_l'] * water_props['c_l'])
print(f"Thermal diffusivity (D) for water: {D_water:.3e} m²/s")
print(f"Thermal diffusivity (D) for sodium: {D_sodium:.3e} m²/s")

# Functions to calculate dimensionless parameters alpha and mu
def calculate_alpha(p_v, p_inf, sigma, rho_l):
    """
    Calculate dimensionless parameter alpha = [p_v(T_∞) - p_∞]³/2 / (2σ(T_∞) ρ_l^½)
    
    Args:
        p_v: Vapor pressure at T_∞ (Pa)
        p_inf: Ambient pressure (Pa)
        sigma: Surface tension (N/m)
        rho_l: Liquid density (kg/m³)
        
    Returns:
        alpha: Dimensionless parameter
    """
    numerator = (p_v - p_inf)**1.5
    denominator = 2 * sigma * rho_l**0.5
    
    # Check if pressure difference is negative (no cavitation)
    if p_v <= p_inf:
        return 0
        
    return numerator / denominator

def calculate_mu(D, sigma, rho_v, L, T_inf, T_sat, p_v, p_inf, lambda_l):
    """
    Calculate dimensionless parameter μ = (1/3) * (2σD/π)^½ * ρ_v * L/k * (T_∞ - T_b)^-1 * {ρ[p_v(T_∞) - p_∞]}^-½
    
    Args:
        D: Thermal diffusivity (m²/s)
        sigma: Surface tension (N/m)
        rho_v: Vapor density (kg/m³)
        L: Latent heat (J/kg)
        T_inf: Bulk liquid temperature (K)
        T_sat: Saturation temperature (K)
        p_v: Vapor pressure at T_∞ (Pa)
        p_inf: Ambient pressure (Pa)
        lambda_l: Thermal conductivity (W/m/K)
        
    Returns:
        mu: Dimensionless parameter
    """
    # Check if temperature difference or pressure difference is too small
    if (T_inf <= T_sat) or (p_v <= p_inf):
        return 0
    
    term1 = (1/3) * ((2*sigma*D)/math.pi)**0.5
    term2 = rho_v * L / lambda_l  # L/k where k is lambda_l (thermal conductivity)
    term3 = 1 / (T_inf - T_sat)
    term4 = 1 / (rho_v * (p_v - p_inf))**0.5
    
    return term1 * term2 * term3 * term4

# Lists to store data for plotting
fluid_types = []
case_names = []
superheats = []
alphas = []
mus = []

# Process each fluid type and its cases
for fluid_type, fluid_data in fluids.items():
    # Get the correct thermal diffusivity based on fluid type
    D = D_sodium if fluid_type == 'sodium' else D_water
    
    # Process cases for current fluid
    for case in fluid_data['cases']:
        # Extract case properties
        name = case['name']
        Delta_T = case['Delta_T']
        p_inf = case['p_inf']
        
        # Get properties based on fluid type
        if fluid_type == 'water':
            properties = fluid_data['properties'].copy()
            properties['p_inf'] = p_inf
            T_sat = properties['T_sat']
        else:  # sodium
            properties = case['properties'].copy() 
            properties['p_inf'] = p_inf
            T_sat = properties['T_sat']
        
        # Calculate superheat temperature T_∞
        T_inf = T_sat + Delta_T
        
        # Extract fluid properties
        R_gas = properties['R_gas']
        sigma = properties['sigma']
        rho_l = properties['rho_l']
        lambda_l = properties['lambda_l']  # thermal conductivity k
        
        # Calculate vapor pressure at T_∞ using ideal gas law (same as in final.py)
        # First, estimate vapor density at T_∞ using the saturation pressure at T_∞
        # For simplicity, use p_sat ≈ p_inf at T_sat, and then scale with temperature
        rho_v_sat = p_inf / (R_gas * T_sat)  # Vapor density at saturation
        # Estimate vapor density at T_∞
        rho_v = rho_v_sat * (T_sat / T_inf)
        # Calculate vapor pressure using ideal gas law
        p_v = rho_v * R_gas * T_inf
        
        # Get latent heat at T_∞
        L = get_latent_heat(fluid_type, T_inf)
        
        # Calculate alpha and mu
        alpha = calculate_alpha(p_v, p_inf, sigma, rho_l)
        mu = calculate_mu(D, sigma, rho_v, L, T_inf, T_sat, p_v, p_inf, lambda_l)
        
        # Store results
        fluid_types.append(fluid_type)
        case_names.append(name)
        superheats.append(Delta_T)
        alphas.append(alpha)
        mus.append(mu)
        
        # Print the values
        print(f"\nCase: {name}")
        print(f"  Superheat (ΔT): {Delta_T} K")
        print(f"  Ambient pressure (p_inf): {p_inf/1e5:.3f} atm")
        print(f"  Vapor pressure at T_∞ (p_v): {p_v/1e5:.5f} atm")
        print(f"  Vapor density (rho_v): {rho_v:.5f} kg/m³")
        print(f"  Surface tension (sigma): {sigma:.5f} N/m")
        print(f"  Thermal conductivity (lambda_l): {lambda_l:.5f} W/(m·K)")
        print(f"  Latent heat (L): {L/1e6:.5f} MJ/kg")
        print(f"  Alpha (α): {alpha:.5e}")
        print(f"  Mu (μ): {mu:.5e}")

# Create figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(12, 10))

# Plot Alpha vs. Superheat
water_indices = [i for i, fluid in enumerate(fluid_types) if fluid == 'water']
sodium_indices = [i for i, fluid in enumerate(fluid_types) if fluid == 'sodium']

# Plot data with different markers for each fluid
axs[0].scatter([superheats[i] for i in water_indices], [alphas[i] for i in water_indices], 
              s=100, marker='o', color='blue', label='Water')
axs[0].scatter([superheats[i] for i in sodium_indices], [alphas[i] for i in sodium_indices], 
              s=100, marker='^', color='red', label='Sodium')

# Add case labels
for i in range(len(case_names)):
    axs[0].annotate(case_names[i], (superheats[i], alphas[i]), 
                   xytext=(5, 5), textcoords='offset points')

axs[0].set_xlabel('Superheat ΔT (K)', fontsize=12)
axs[0].set_ylabel('α = [pv(T∞) - p∞]³/²/2σ(T∞)ρ^½', fontsize=12)
axs[0].set_title('Dimensionless Parameter α vs. Superheat', fontsize=14)
axs[0].grid(True)
axs[0].legend()

# Plot Mu vs. Superheat
axs[1].scatter([superheats[i] for i in water_indices], [mus[i] for i in water_indices], 
              s=100, marker='o', color='blue', label='Water')
axs[1].scatter([superheats[i] for i in sodium_indices], [mus[i] for i in sodium_indices], 
              s=100, marker='^', color='red', label='Sodium')

# Add case labels
for i in range(len(case_names)):
    axs[1].annotate(case_names[i], (superheats[i], mus[i]), 
                   xytext=(5, 5), textcoords='offset points')

axs[1].set_xlabel('Superheat ΔT (K)', fontsize=12)
axs[1].set_ylabel('μ = (1/3)(2σD/π)^½ ρv(L/k)(T∞-Tb)^-1{ρ[pv(T∞)-p∞]}^-½', fontsize=12)
axs[1].set_title('Dimensionless Parameter μ vs. Superheat', fontsize=14)
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.savefig('dimensionless_parameters.png', dpi=300)

# Create a table with all the calculated values
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')
ax.axis('tight')

# Create table data
table_data = []
table_data.append(['Case', 'Superheat (K)', 'Alpha (α)', 'Mu (μ)'])
for i in range(len(case_names)):
    table_data.append([case_names[i], f"{superheats[i]}", f"{alphas[i]:.5e}", f"{mus[i]:.5e}"])

table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)
plt.title('Dimensionless Parameters for All Test Cases', fontsize=16)
plt.tight_layout()
plt.savefig('dimensionless_parameters_table.png', dpi=300)

plt.show()