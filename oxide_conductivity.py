import rfgen as rf
import numpy as np
from scipy.special import erfcinv
from Greenwood_Tripp_Model import compute_holm_radius, F1_analytical
import os

# Load results from Greenwood-Tripp model
try:
    data = np.load("Results_ind_type_sphere_full_history.npz")
    print("Loaded simulation results.")
    
    # Overwrite parameters with loaded values
    eta = float(data['eta'])
    beta = float(data['beta'])
    Radius = float(data['Radius'])
    sigma = float(data['sigma'])
    Estar = float(data['Estar'])
    
except FileNotFoundError:
    print("Results file not found. Using default parameters.")
    # Roughness parameters
    eta = 2e8           # Asperity density (1/m²)
    beta = 0.3e-4       # Asperity tip radius (m)
    # Indenter geometry
    Radius = 0.01           # Indenter radius (m)
    sigma = 20e-6
    E = 2.1e11              # Young's modulus (Pa)
    nu = 0.3                # Poisson's ratio
    Estar = E/(2*(1 - nu**2))

# Compute Resistance

D = data['d']
Force = data['force']
P = data['p']
W = data['w']
H = data['h']
r = data['r']
if 'a_computed' in data: A_computed = data['a_computed']
if 'holm_radius' in data: Holm_radius = data['holm_radius']

dx = np.sqrt(1/eta)

size = 0.4 * Radius
x_lim = size / 2.

x_coarse = np.arange(-x_lim, x_lim+dx, dx)
y_coarse = np.arange(-x_lim, x_lim+dx, dx)
Xc, Yc = np.meshgrid(x_coarse, y_coarse, indexing='ij')

# Pre-calculate Ac for all steps to avoid recomputing in inner loops
print("Pre-calculating contact areas...")
Ac_stack = np.zeros((H.shape[0], Xc.shape[0], Xc.shape[1]))
Resistance_self = np.zeros_like(D)
Resistance_inter = np.zeros_like(D)

for k in range(H.shape[0]):        
    area_fraction = np.pi * beta * eta * F1_analytical(H[k], sigma)
    sum_radii = 0.0
    for i in range(Xc.shape[0]):
        for j in range(Xc.shape[1]):
            polar_r = np.sqrt(Xc[i,j]**2 + Yc[i,j]**2)
            area_frac = np.interp(polar_r, r, area_fraction, left=0.0, right=0.0)
            contact_area_total = area_frac * dx**2
            Ac_stack[k, i, j] = contact_area_total
            if contact_area_total > 0:
                sum_radii += np.sqrt(contact_area_total/np.pi)
    
    R_spots = 1.0 / (2.0 * sum_radii) if sum_radii > 0 else np.inf
    R_Holm = 1.0 / (2.0 * Holm_radius[k]) if Holm_radius[k] > 0 else np.inf
    
    Resistance_self[k] = R_spots
    Resistance_inter[k] = R_Holm

# Oxide correlation length
L_all = np.array([5,10,20]) / eta**0.5
Hurst = 0.5

# Oxide fraction
Fract = [0.3, 0.5, 0.7]

# Resistivity
resistivity = 2.7e-8  # Ohm meter (Al)

N = 1024
Nsurfaces = 100

# Indices for resampling
scale_x = N / Xc.shape[0]
scale_y = N / Xc.shape[1]
idx_i = (np.arange(Xc.shape[0]) * scale_x).astype(int)
idx_j = (np.arange(Xc.shape[1]) * scale_y).astype(int)
II, JJ = np.meshgrid(idx_i, idx_j, indexing='ij')

for L_idx, L in enumerate(L_all):
    kl = size / (2*L) 
    ks = 3 * kl
    for fract_idx, fract in enumerate(Fract):
        print(f"Simulating L={L:.2e}, fract={fract}")
        
        # Generate rough surface
        Oxide_maps = np.zeros((Nsurfaces, N, N), dtype=bool)
        for j in range(Nsurfaces):
            rng = np.random.default_rng(42+j)
            surface = rf.selfaffine_field(
                dim=2, 
                N=N,
                Hurst=Hurst,
                k_low=kl/N,
                k_high=ks/N,
                plateau=False,
                noise=False,
                rng=rng)

            # Normalize to standard deviation of 1
            current_rms = np.std(surface)
            surface /= current_rms
            G0 = np.sqrt(2) * erfcinv(2*fract)
            oxide = surface > G0

            Oxide_maps[j] = oxide

        Oxidized_Resistance = np.zeros((H.shape[0], Nsurfaces))
        
        for k in range(H.shape[0]):
            Ac = Ac_stack[k]
            
            for j in range(Nsurfaces):
                oxide = Oxide_maps[j]
                
                # Vectorized resampling of oxide map
                oxide_resampled = oxide[II, JJ]
                
                # Element-wise multiplication
                Ac_oxidized = Ac * (1 - oxide_resampled)
                
                # Vectorized sum of radii
                valid_mask = Ac_oxidized > 0
                sum_radii_oxide = np.sum(np.sqrt(Ac_oxidized[valid_mask] / np.pi))

                R_spots_oxide = 1.0 / (2.0 * sum_radii_oxide) if sum_radii_oxide > 0 else np.inf
                GreenwoodHolmRadius = compute_holm_radius(Xc, Yc, Ac_oxidized)
                R_Holm_oxide = 1.0 / (2.0 * GreenwoodHolmRadius) if GreenwoodHolmRadius > 0 else np.inf

                Oxidized_Resistance[k,j] = R_spots_oxide + R_Holm_oxide

        # Save results
        filename = f"Oxide_Results_L_{L_idx}_fract_{fract_idx}.npz"
        np.savez(filename, 
                 Oxidized_Resistance=Oxidized_Resistance,
                 Resistance_self=Resistance_self,
                 Resistance_inter=Resistance_inter,
                 Force=Force,
                 L=L,
                 fract=fract,
                 eta=eta,
                 beta=beta,
                 Radius=Radius,
                 sigma=sigma,
                 Estar=Estar,
                 resistivity=resistivity,
                 L_idx=L_idx,
                 fract_idx=fract_idx
                 )
        print(f"Saved {filename}")
