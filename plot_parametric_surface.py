import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# LaTeX-style plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 8})

# Constants for normalization
rho_star = 1.0       # TODO: User to provide
E_star = 1.1538e11   # E/(2*(1-nu^2)) with E=2.1e11, nu=0.3
F0 = 1.0             # TODO: User to provide
R_star = 0.01        # Indenter radius
sigma = 20e-6        # RMS roughness

def poly2_2_mult(X, c0, c1, c2, c3, c4):
    x, y = X
    # Multiplicative decomposition: (c0 + c1*x + c2*x^2) * (1 + c3*y + c4*y^2)
    return (c0 + c1 * x**2 + c2 * x**8) * (1 + c3*y + c4*y**2)

def fit_and_report(xi, lam, z, name):
    # Flatten data for fitting
    X_flat = np.vstack((xi.ravel(), lam.ravel()))
    z_flat = z.ravel()
    
    # Initial guess: c0=mean(z), others 0
    p0 = [np.mean(z_flat), 0, 0, 0, 0]
    
    popt, pcov = curve_fit(poly2_2_mult, X_flat, z_flat, p0=p0)
    
    print(f"--- Fitting results for {name} ---")
    print(f"Model: f(xi, lambda) = (c0 + c1*xi^2 + c2*xi^8) * (1 + c3*lambda + c4*lambda^2)")
    print(f"Coefficients:")
    print(f"c0 = {popt[0]:.6e}")
    print(f"c1 = {popt[1]:.6e}")
    print(f"c2 = {popt[2]:.6e}")
    print(f"c3 = {popt[3]:.6e}")
    print(f"c4 = {popt[4]:.6e}")
    
    # Calculate R-squared
    residuals = z_flat - poly2_2_mult(X_flat, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((z_flat - np.mean(z_flat))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"R-squared: {r_squared:.6f}\n")
    
    return popt

def main():
    # Load data
    df_orig = pd.read_csv('fitting_constants_parametric.csv', skipinitialspace=True)
    df_focus = pd.read_csv('focus_fitting_constants_parametric.csv', skipinitialspace=True)
    
    # Merge and prioritize focus data
    df = pd.concat([df_orig, df_focus])
    
    # Round coordinates to avoid floating point mismatches
    df['oxide_fraction'] = df['oxide_fraction'].round(6)
    df['correlation_sqrt_eta'] = df['correlation_sqrt_eta'].round(6)
    
    # Drop duplicates based on coordinates, keeping the last one (which comes from focus)
    df = df.drop_duplicates(subset=['oxide_fraction', 'correlation_sqrt_eta'], keep='last')
    
    # Normalize alpha
    # alpha = (alpha_0 / (2*rho_star)) * (E_star / (2*F0))**gamma * R_star**((1+gamma)/2) * sigma**((1+3*gamma)/2)
    # Here 'a' is alpha_0 and 'b' is gamma
    df['alpha_norm'] = (df['a'] / (2 * rho_star)) * \
                       (E_star / (2 * F0)) ** df['b'] * \
                       R_star ** ((1 + df['b']) / 2) * \
                       sigma ** ((1 + 3 * df['b']) / 2)
    
    # Extract unique values to form the grid
    xi_vals = np.sort(df['oxide_fraction'].unique())
    lam_vals = np.sort(df['correlation_sqrt_eta'].unique())
    
    # Create meshgrid for plotting and fitting
    XI, LAM = np.meshgrid(xi_vals, lam_vals)
    
    # Reshape data
    # The data in CSV might not be sorted, so we pivot
    pivot_a = df.pivot(index='correlation_sqrt_eta', columns='oxide_fraction', values='alpha_norm')
    pivot_b = df.pivot(index='correlation_sqrt_eta', columns='oxide_fraction', values='b')
    
    # Ensure the pivot table is sorted by index and columns
    pivot_a = pivot_a.sort_index().sort_index(axis=1)
    pivot_b = pivot_b.sort_index().sort_index(axis=1)
    
    Z_a = pivot_a.values
    Z_b = pivot_b.values
    
    # Perform fitting
    fit_and_report(XI, LAM, Z_a, 'alpha')
    fit_and_report(XI, LAM, Z_b, 'gamma')
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(6.1, 3.3))
    
    # Calculate extent for imshow to center pixels on grid points
    dx = xi_vals[1] - xi_vals[0]
    dy = lam_vals[1] - lam_vals[0]
    extent = [xi_vals[0] - dx/2, xi_vals[-1] + dx/2, lam_vals[0] - dy/2, lam_vals[-1] + dy/2]
    
    # Plot a
    im_a = axes[0].imshow(Z_a, origin='lower', aspect='auto', extent=extent, cmap='viridis')
    axes[0].set_title('Parameter $\\alpha$')
    axes[0].set_xlabel(r'$\xi$ (Oxide Fraction)')
    axes[0].set_ylabel(r'$l_{\mathcal{O}}\sqrt{\eta}$ (Normalized Correlation Length)')
    fig.colorbar(im_a, ax=axes[0])
    
    # Plot b
    im_b = axes[1].imshow(Z_b, origin='lower', aspect='auto', extent=extent, cmap='viridis')
    axes[1].set_title('Parameter $\\gamma$')
    axes[1].set_xlabel(r'$\xi$ (Oxide Fraction)')
    axes[1].set_ylabel(r'$l_{\mathcal{O}}\sqrt{\eta}$ (Normalized Correlation Length)')
    fig.colorbar(im_b, ax=axes[1])
    
    plt.tight_layout()
    fig.savefig('fitting_parameters_surface.pdf', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
