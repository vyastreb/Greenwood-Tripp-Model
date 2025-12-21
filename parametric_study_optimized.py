import os
# Set Numba to single thread to avoid oversubscription when using joblib
os.environ['NUMBA_NUM_THREADS'] = '1'

import sys
# sys.path.append("")

import rfgen as rf
import numpy as np
from scipy.special import erfcinv
from Greenwood_Tripp_Model import compute_holm_radius, F1_analytical
from joblib import Parallel, delayed
import time
from numba import njit, prange

@njit(parallel=True)
def _precompute_ac_stack_numba(Xc, Yc, r, area_fraction_stack, dx):
    n_steps = area_fraction_stack.shape[0]
    nx, ny = Xc.shape
    Ac_stack = np.zeros((n_steps, nx, ny))
    sum_radii_arr = np.zeros(n_steps)
    
    for k in prange(n_steps):
        area_fraction = area_fraction_stack[k]
        sum_radii = 0.0
        for i in range(nx):
            for j in range(ny):
                polar_r = np.sqrt(Xc[i,j]**2 + Yc[i,j]**2)
                # np.interp with left/right handling
                area_frac = np.interp(polar_r, r, area_fraction)
                
                if polar_r < r[0] or polar_r > r[-1]:
                     area_frac = 0.0
                
                contact_area_total = area_frac * dx**2
                Ac_stack[k, i, j] = contact_area_total
                if contact_area_total > 0:
                    sum_radii += np.sqrt(contact_area_total/np.pi)
        sum_radii_arr[k] = sum_radii
        
    return Ac_stack, sum_radii_arr

@njit
def _compute_resistance_loop(Ac_stack, oxide_resampled, Xc, Yc):
    n_steps = Ac_stack.shape[0]
    R_spots_oxide_arr = np.zeros(n_steps)
    R_Holm_oxide_arr = np.zeros(n_steps)
    
    for k in range(n_steps):
        Ac = Ac_stack[k]
        
        # We need Ac_oxidized for compute_holm_radius
        Ac_oxidized = np.empty_like(Ac)
        sum_radii_oxide = 0.0
        
        nx, ny = Ac.shape
        for i in range(nx):
            for j in range(ny):
                val = Ac[i, j] * (1.0 - oxide_resampled[i, j])
                Ac_oxidized[i, j] = val
                if val > 0:
                    sum_radii_oxide += np.sqrt(val / np.pi)
        
        R_spots_oxide = 1.0 / (2.0 * sum_radii_oxide) if sum_radii_oxide > 0 else np.inf
        
        # Compute Holm radius
        GreenwoodHolmRadius = compute_holm_radius(Xc, Yc, Ac_oxidized)
        R_Holm_oxide = 1.0 / (2.0 * GreenwoodHolmRadius) if GreenwoodHolmRadius > 0 else np.inf
        
        R_spots_oxide_arr[k] = R_spots_oxide
        R_Holm_oxide_arr[k] = R_Holm_oxide
        
    return R_spots_oxide_arr + R_Holm_oxide_arr

def compute_resistance_single_surface(seed, L, fract, Ac_stack, Xc, Yc, II, JJ, N, Hurst, kl, ks):
    """
    Generates a surface and computes resistance for all load steps.
    """
    rng = np.random.default_rng(seed)
    # Generate rough surface
    # Note: Assuming rf.selfaffine_field signature matches the one in parametric_study_oxide_conductivity.py
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
    if current_rms > 0:
        surface /= current_rms
    
    G0 = np.sqrt(2) * erfcinv(2*fract)
    oxide = surface > G0
    
    # Vectorized resampling of oxide map
    oxide_resampled = oxide[II, JJ].astype(np.float64)
    
    return _compute_resistance_loop(Ac_stack, oxide_resampled, Xc, Yc)

def main():
    base_path = "/home/users02/vyastrebov/ARTICLES/2025/07_Oxides_in_Contact/Article/Code/GreenwoodTripp/"
    results_file = os.path.join(base_path, "Results_ind_type_sphere_full_history.npz")
    
    # Load results from Greenwood-Tripp model
    try:
        data = np.load(results_file)
        print(f"Loaded simulation results from {results_file}")
        
        # Overwrite parameters with loaded values
        eta = float(data['eta'])
        beta = float(data['beta'])
        Radius = float(data['Radius'])
        sigma = float(data['sigma'])
        Estar = float(data['Estar'])
        
        D = data['d']
        Force = data['force']
        H = data['h']
        r = data['r']
        if 'holm_radius' in data: 
            Holm_radius = data['holm_radius']
        else:
            # Fallback if not present (should be there based on read file)
            Holm_radius = np.zeros_like(D) 
        
    except FileNotFoundError:
        print(f"Results file {results_file} not found. Please run the Greenwood-Tripp model first.")
        return

    # Compute Resistance for clean contact
    dx = np.sqrt(1/eta)
    size = 0.4 * Radius
    x_lim = size / 2.

    x_coarse = np.arange(-x_lim, x_lim+dx, dx)
    y_coarse = np.arange(-x_lim, x_lim+dx, dx)
    Xc, Yc = np.meshgrid(x_coarse, y_coarse, indexing='ij')

    # Pre-calculate Ac for all steps
    print("Pre-calculating contact areas...")
    
    # Precompute area fractions
    area_fraction_stack = np.zeros((H.shape[0], r.shape[0]))
    for k in range(H.shape[0]):
        area_fraction_stack[k] = np.pi * beta * eta * F1_analytical(H[k], sigma)
        
    Ac_stack, sum_radii_arr = _precompute_ac_stack_numba(Xc, Yc, r, area_fraction_stack, dx)
    
    Resistance_self = np.zeros_like(D)
    Resistance_inter = np.zeros_like(D)
    
    for k in range(H.shape[0]):
        sum_radii = sum_radii_arr[k]
        R_spots = 1.0 / (2.0 * sum_radii) if sum_radii > 0 else np.inf
        R_Holm = 1.0 / (2.0 * Holm_radius[k]) if Holm_radius[k] > 0 else np.inf
        
        Resistance_self[k] = R_spots
        Resistance_inter[k] = R_Holm

    # Parameters for study
    # l*sqrt(eta) in (1, 30)
    # User requested at least 10 values
    L_sqrt_eta_values = np.linspace(1, 19, 10)
    L_all = L_sqrt_eta_values / np.sqrt(eta)
    
    # Oxide fraction in (0.1, 0.9)
    # User requested at least 10 values
    Fract_values = np.linspace(0.1, 0.8, 8)
    
    Hurst = 0.5
    resistivity = 2.7e-8  # Ohm meter (Al)
    N = 1024
    Nsurfaces = 200 

    # Indices for resampling
    scale_x = N / Xc.shape[0]
    scale_y = N / Xc.shape[1]
    idx_i = (np.arange(Xc.shape[0]) * scale_x).astype(int)
    idx_j = (np.arange(Xc.shape[1]) * scale_y).astype(int)
    II, JJ = np.meshgrid(idx_i, idx_j, indexing='ij')

    # Prepare tasks
    tasks = []
    for L_idx, L in enumerate(L_all):
        kl = size / (2*L) 
        ks = 3 * kl
        for fract_idx, fract in enumerate(Fract_values):
            for j in range(Nsurfaces):
                seed = 42 + j + L_idx*1000 + fract_idx*100000 # Unique seed
                tasks.append((seed, L, fract, L_idx, fract_idx, j, kl, ks))

    print(f"Total tasks: {len(tasks)}")
    print("Starting parallel simulation...")
    
    start_time = time.time()
    # Run parallel
    # n_jobs=-1 uses all available cores
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(compute_resistance_single_surface)(
            seed, L, fract, Ac_stack, Xc, Yc, II, JJ, N, Hurst, kl, ks
        ) for seed, L, fract, L_idx, fract_idx, j, kl, ks in tasks
    )
    end_time = time.time()
    print(f"Simulation finished in {end_time - start_time:.2f} seconds.")
    
    # Organize results
    # Shape: (n_L, n_Fract, n_surfaces, n_steps)
    Oxidized_Resistance_All = np.zeros((len(L_all), len(Fract_values), Nsurfaces, H.shape[0]))
    
    for idx, res in enumerate(results):
        seed, L, fract, L_idx, fract_idx, j, kl, ks = tasks[idx]
        Oxidized_Resistance_All[L_idx, fract_idx, j, :] = res

    # Save single NPZ
    output_npz = "Oxide_Parametric_Study_Results.npz"
    print(f"Saving all results to '{output_npz}'...")
    np.savez(output_npz,
             Oxidized_Resistance_All=Oxidized_Resistance_All,
             Resistance_self=Resistance_self,
             Resistance_inter=Resistance_inter,
             Force=Force,
             L_values=L_all,
             Fract_values=Fract_values,
             eta=eta,
             beta=beta,
             Radius=Radius,
             sigma=sigma,
             Estar=Estar,
             resistivity=resistivity,
             L_sqrt_eta_values=L_sqrt_eta_values
             )

    # Fit exponents and save to CSV
    print("Fitting exponents...")
    output_csv = "fitting_constants_parametric.csv"
    with open(output_csv, "w") as f:
        f.write("oxide_fraction, correlation_sqrt_eta, a, b\n")
        
        force_min_fit = 200.0
        mask_fit = Force >= force_min_fit
        
        if np.sum(mask_fit) > 1:
            F_fit = Force[mask_fit]
            
            for L_idx, L in enumerate(L_all):
                for fract_idx, fract in enumerate(Fract_values):
                    # Mean resistance across surfaces
                    R_ox_all_surfaces = Oxidized_Resistance_All[L_idx, fract_idx, :, :] # (Nsurfaces, n_steps)
                    R_ox_mean = np.mean(resistivity * R_ox_all_surfaces, axis=0) # (n_steps,)
                    
                    R_ox_fit_data = R_ox_mean[mask_fit]
                    
                    # Check if we have valid data for fitting
                    if len(R_ox_fit_data) > 0 and np.all(R_ox_fit_data > 0):
                        coeffs_ox = np.polyfit(np.log(F_fit), np.log(R_ox_fit_data), 1)
                        b_ox = -coeffs_ox[0]
                        a_ox = np.exp(coeffs_ox[1])
                        
                        f.write(f"{fract}, {L * np.sqrt(eta)}, {a_ox}, {b_ox}\n")

    print(f"Fitting constants saved to '{output_csv}'.")
    print("Done.")

if __name__ == "__main__":
    main()
