"""
Greenwood-Tripp contact model for rough surfaces [A]
Implements multi-asperity contact theory with iterative pressure-displacement coupling

References:
+ [A] Greenwood, J.A. and Tripp, J.H., The Elastic Contact of Rough Spheres, 
      Journal of Applied Mechanics, 34(1), p.153-159 (1967).
+ [B] Johnson, K.L., Contact Mechanics, Cambridge University Press (1985).

**Remarks**

+ Eq. (4) in [A] seems to be incorrect. An appropriate equation from Johnson's book [B] is used instead (integral of Eq. (3.96a) in [B]).

Author: Vladislav A. Yastrebov (Mines Paris - PSL, CNRS)
Date: September-December 2025
License: BSD 3-Clause License
AI usage: Claude Sonnet 4.5 for final cleanup and optimization, the core code is written by the author.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.special import ellipk,erfc
from numba import njit, prange
from matplotlib.patches import CirclePolygon
from matplotlib.collections import PatchCollection

# LaTeX-style plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 10})


@njit(parallel=True)
def compute_holm_radius(Xc, Yc, Ac):
    """Compute Holm's effective contact radius using parallel computation"""
    numerator = 0.0
    denominator = 0.0
    n, m = Xc.shape
    
    for i in prange(n):
        for j in prange(m):
            denominator += Ac[i, j]
            for k in prange(n):
                for l in prange(m):
                    if not (i == k and j == l):
                        dist = np.sqrt((Xc[i, j] - Xc[k, l])**2 + (Yc[i, j] - Yc[k, l])**2)
                        numerator += Ac[i, j] * Ac[k, l] / dist
    
    alpha_inv = numerator / (denominator**2 * np.pi)
    return 1.0 / alpha_inv / 2.0


def indenter_shape(r, indenter_type, params):
    """
    Define indenter geometry
    
    Args:
        r: radial coordinate
        indenter_type: 'sphere', 'cone', or 'flat'
        params: geometry parameters [radius] or [radius, fillet_radius]
    """
    r = np.asarray(r)
    
    if indenter_type == 'sphere':
        R = params[0]
        return r**2 / (2 * R)
    
    elif indenter_type == 'cone':
        alpha = params[0]
        return r * np.tan(alpha)
    
    elif indenter_type == 'flat':
        R_flat = params[0]
        fillet_radius = params[1] if len(params) > 1 else 0.0
        
        if fillet_radius > 0:
            z = np.zeros_like(r)
            mask1 = r <= R_flat - fillet_radius
            z[mask1] = 0.0
            
            mask2 = (r > R_flat - fillet_radius) & (r < R_flat)
            r_local = r[mask2] - R_flat + fillet_radius
            z[mask2] = fillet_radius - np.sqrt(fillet_radius**2 - r_local**2)
            
            mask3 = r >= R_flat
            z[mask3] = 1.0
            return z
        else:
            return np.where(r <= R_flat, 0.0, 1.0)
    
    else:
        raise ValueError("Unsupported indenter type. Use 'sphere', 'cone', or 'flat'.")


def probability_density(x, sigma):
    """Gaussian probability density function for surface heights"""
    return np.exp(-x**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)


def F_integral(u, sigma, power):
    """
    Compute F_n integral for Greenwood-Tripp model
    Integral from u to infinity of (z-u)^n * phi(z) dz
    
    Args:
        u: separation distance
        sigma: RMS roughness
        power: exponent (1.5 for F_{3/2}, 1.0 for F_1)
    """
    cutoff_factor = 20
    if u > cutoff_factor * sigma:
        return 0.0
    
    integrand = lambda z: (z - u)**power * probability_density(z, sigma)
    result, _ = integrate.quad(integrand, u, u + cutoff_factor * sigma, epsabs=1e-8, epsrel=1e-8)
    return result


def F32(h, sigma):
    """Vectorized F_{3/2} integral"""
    if np.isscalar(h):
        return F_integral(h, sigma, 1.5)
    return np.array([F_integral(hi, sigma, 1.5) for hi in np.atleast_1d(h)])


def F1(h, sigma):
    """Vectorized F_1 integral"""
    if np.isscalar(h):
        return F_integral(h, sigma, 1.0)
    return np.array([F_integral(hi, sigma, 1.0) for hi in np.atleast_1d(h)])


def integrand(x, rv, p_interpol):
    """
    Integrand for Boussinesq displacement solution
    See Johnson [B], Eq. (3.96a)
    """
    eps = 1e-10
    k = 4 * x * rv / (rv + x + eps)**2
    return x / (x + rv + eps) * p_interpol(x) * ellipk(k)


def get_displacement_from_p(p_vals, r_grid, Estar, integration_limit):
    """Compute surface displacement from pressure using Boussinesq solution"""
    r = np.asarray(r_grid)
    coef = 4.0 / (np.pi * Estar)
    result = np.zeros_like(r, dtype=float)
    
    p_interp = lambda x: np.interp(x, r, p_vals, left=0.0, right=0.0)
    
    for i, rv in enumerate(r):
        val, _ = integrate.quad(
            lambda x: integrand(x, rv, p_interp),
            1e-9, integration_limit,
            epsabs=1e-10, epsrel=1e-8,
            limit=100  # Increase subdivision limit
        )
        result[i] = coef * val
    
    return result if r.ndim > 0 else result[0]


def plot_state(r, p, z, w, Radius, sigma, pstar, title="Current state", filename=None, set_limits=False):
    """Visualize current solution state"""
    fig, ax = plt.subplots(2, 2, figsize=(5.61, 5.61))
    # fig, ax = plt.subplots(2, 2, figsize=(10, 3))
    # fig.suptitle(title)

    # Compute associated Hertz solution
    force_computed = 2 * np.pi * integrate.simpson(p * r, x=r)
    a_computed = ((3 * force_computed * Radius) / (4 * Estar))**(1/3)
    p0 = (3 * force_computed) / (2 * np.pi * a_computed**2)
    ref_r = np.linspace(0, a_computed, 100)
    ref_pressure = p0 * np.sqrt(1 - (ref_r / a_computed)**2)

    # Prolongate Hertzian displacement for comparison
    r_prolongation = np.concatenate([ref_r, np.linspace(a_computed, np.max(r), 100)])
    hertz_pressure = np.concatenate([ref_pressure, np.zeros(100)])
    hertz_disp =  get_displacement_from_p(hertz_pressure, r_prolongation, Estar, np.max(r))

    # Pressure distribution
    ax[0,0].grid()
    ax[0,0].plot(r/Radius, p/pstar, "-", label="Rough")
    ax[0,0].plot(ref_r/Radius, ref_pressure/pstar, "r--", label="Hertzian")
    ax[0,0].set_ylabel(r"Normalized pressure, $p/\bar{p}$")
    ax[0,0].set_xlabel(r"Normalized radial coordinate, $\frac r R$")
    if set_limits:
        ax[0,0].set_xlim(0, np.max(r)/Radius)
        ax[0,0].set_ylim(0, max(np.max(p), np.max(ref_pressure))*1.1/pstar)
    ax[0,0].legend()
    
    # Surface configuration
    ax[0,1].grid()
    if set_limits:
        ax[0,1].set_xlim(0, np.max(r)/Radius)
        ax[0,1].set_ylim(-2, 10)
    ax[0,1].axhline(0, color='grey', linestyle='--', alpha=1.0, label="__")
    ax[0,1].plot(r/Radius, z/sigma, "--", color="k", label="Reference")
    ax[0,1].plot(r/Radius, (z+w)/sigma, "-", color="#1f77b4", label="Rough", 
    zorder=3)
    def_hertz = r_prolongation**2/(2*Radius) - 2*sigma + hertz_disp
    ax[0,1].plot(r_prolongation/Radius, def_hertz/sigma, "r--", label="Hertzian", zorder=2)
    # for mult, alpha in [(1, 0.5), (2, 0.25), (3, 0.125)]:
    #     ax[0,1].axhline(mult, color='grey', linestyle='--', alpha=alpha)
    ax[0,1].set_ylabel(r"Normalized configuration, $z/\sigma$")
    ax[0,1].set_xlabel(r"Normalized radial coordinate, $\frac r R$")
    ax[0,1].legend()
    
    # Displacement
    ax[1,0].grid()
    if set_limits:
        ax[1,0].set_xlim(0, np.max(r)/Radius)
        ax[1,0].set_ylim(0, 3.5)
    ax[1,0].plot(r/Radius, w/sigma, label="Rough")
    ax[1,0].plot(r_prolongation/Radius, hertz_disp/sigma, "r--", label="Hertzian")

    ax[1,0].set_ylabel(r"Normalized displacement, $u/\sigma$")
    ax[1,0].set_xlabel(r"Normalized radial coordinate, $\frac r R$")
    ax[1,0].legend()
    
    # Contact area fraction
    area_fraction = chi * F1(z + w, sigma)

    delta = 2 * sigma
    zp = (r**2/(2*Radius) - delta)/sigma
    asymptotic_area_fraction =  chi * (sigma / np.sqrt(2*np.pi) * np.exp(- r**4/(8 * sigma**2 * Radius**2)) * np.exp(r**2 * delta/(2*Radius*sigma**2)) * np.exp(-delta**2/(2*sigma**2))  - 0.5 * sigma * zp * erfc(zp / np.sqrt(2))) #* np.exp(r**2 * delta/(2*Radius*sigma**2)) * np.exp(-delta**2/(2*sigma**2))

    asymptotic_area_fraction =  chi * sigma / np.sqrt(2*np.pi) * np.exp(- r**4/(8 * sigma**2 * Radius**2))
    asymptotic_area_fraction =  chi * (sigma / np.sqrt(2*np.pi) * np.exp(- zp**2 / 2) - 0.5 * sigma * zp * erfc(zp / np.sqrt(2)))

    asymptotic_area_fraction[r/Radius < 0.1] = np.nan

    ax[1,1].grid()
    if set_limits:
        ax[1,1].set_xlim(0, np.max(r)/Radius)
        ax[1,1].set_ylim(1e-10, np.max(area_fraction) * 10)
    ax[1,1].set_yscale('log')
    ax[1,1].plot(r/Radius, area_fraction, color="green", label="Rough")
    ax[1,1].plot(r/Radius, asymptotic_area_fraction, "k--", label="Asymptotic")
    ax[1,1].set_ylabel("Contact area fraction")
    ax[1,1].set_xlabel(r"Normalized radial coordinate, $\frac r R$")
    ax[1,1].legend()

    # Add subplot labels
    letters = ['(a)', '(b)', '(c)', '(d)']
    for i in range(2):
        for j in range(2):
            ax[i, j].text(-0.1, 1.05, letters[i*2+j], transform=ax[i, j].transAxes, fontweight='bold')

    plt.tight_layout()
    if filename:
        fig.savefig(filename, bbox_inches='tight', pad_inches=0)
    return fig


# =============================================================================
#         PARAMETERS
# =============================================================================

# Material properties
E = 2.1e11              # Young's modulus (Pa)
nu = 0.3                # Poisson's ratio
Estar = E / (2 * (1 - nu**2))

# Roughness parameters
sigma = 20e-6            # RMS roughness (m)
eta = 2e2 * 1e6         # Asperity density (1/m²)
beta = 0.3e-4           # Asperity tip radius (m)
mu = (4.0/3.0) * eta * Estar * np.sqrt(beta)
chi = np.pi * eta * beta

# Indenter geometry
ind_type = 'sphere'
Radius = 0.01           # Indenter radius (m)

# Numerical parameters
d = -2 * sigma          # Initial separation (m)
pstar = Estar * np.sqrt(sigma / Radius)
Np = 100                 # Grid points
kappa = 0.3              # Relaxation factor
tolerance = 1e-3         # Convergence tolerance
max_iter = 100

# =============================================================================
#       SETUP
# =============================================================================

if ind_type == 'sphere':
    params = [Radius]
    integration_limit = 0.3 * Radius
elif ind_type == 'flat':
    fillet_radius = 0.05
    params = [Radius, fillet_radius]
    integration_limit = 2.0 * Radius
elif ind_type == 'cone':
    params = [np.pi/6]  # 30 degree half-angle
    integration_limit = 0.1 * Radius

r = np.linspace(0, integration_limit, Np)
w = np.zeros_like(r)
w_old = np.zeros_like(r)

# Initial configuration
w0 = d + indenter_shape(r, ind_type, params)
w = w0.copy()
p = mu * F32(w, sigma)

print(f"Initial max pressure: {np.max(p):.2e} Pa")

w_new = get_displacement_from_p(p, r, Estar, integration_limit)
plot_state(r, p, w0, w_new, Radius, sigma, pstar, "Initial state")

# =============================================================================
#       ITERATIVE SOLVER
# =============================================================================

h = d + indenter_shape(r, ind_type, params)
eps_old = 1e10

print("\nStarting iterations...")
for it in range(max_iter):
    w_new = get_displacement_from_p(p, r, Estar, integration_limit)
    w = kappa * w_new + (1 - kappa) * w_old
    h += (w - w_old)
    p = mu * F32(h, sigma)
    
    eps = np.max(np.abs(w - w_old) / kappa) / (np.max(np.abs(w_old)) + 1e-20)
    w_old = w.copy()
    
    if it > 2 and eps > eps_old * 1.01:
        kappa = max(kappa * 0.5, 0.01)
        print(f"\nReducing relaxation to {kappa:.3f}")
    eps_old = eps
    
    print(f"\rIteration {it+1:03d}/{max_iter}, Error: {eps:.2e}", end='', flush=True)
    
    # Since we use relaxation, we check convergence based on the adjusted tolerance
    if eps < kappa * tolerance:
        print(f"\nConverged in {it+1} iterations (error: {eps:.2e})")
        break
else:
    print(f"\nWarning: Max iterations reached (error: {eps:.2e})")

# =============================================================================
#       RESULTS
# =============================================================================

force_computed = 2 * np.pi * integrate.simpson(p * r, x=r)
print(f"\nComputed force: {force_computed:.3f} N")

if ind_type == 'sphere':
    a_computed = ((3 * force_computed * Radius) / (4 * Estar))**(1/3)
    p0 = (3 * force_computed) / (2 * np.pi * a_computed**2)
    ref_r = np.linspace(0, a_computed, 100)
    ref_pressure = p0 * np.sqrt(1 - (ref_r / a_computed)**2)
    print(f"Roughness parameter sigma*R/a^2: {sigma*Radius/a_computed**2:.3f}")
elif ind_type == 'flat':
    p0 = force_computed / (np.pi * Radius**2)
    ref_r = np.linspace(0, Radius*(1-1e-4), 100)
    ref_pressure = 0.5 * p0 / np.sqrt(1 - ref_r**2 / Radius**2)

area_fraction = chi * F1(h, sigma)

# Plot final pressure and area fraction
fig, ax1 = plt.subplots(figsize=(5.1, 3.))
ax1.plot(ref_r/Radius, ref_pressure/pstar, "r--", linewidth=2, label="Hertzian")
ax1.plot(r/Radius, p/pstar, "-", color="#1f77b4", linewidth=2, label="Greenwood-Tripp")
ax1.set_xlabel(r"Normalized radial coordinate, $r/R$")
ax1.set_ylabel(r"Normalized pressure, $p/\bar{p}$")
ax1.grid(True, alpha=0.3)

if ind_type == 'sphere':
    ax1.set_xlim(0, 3*a_computed/Radius)
    ax1.set_ylim(0, np.max(ref_pressure)*1.1/pstar)
else:
    ax1.set_xlim(0, np.max(r/Radius))
    ax1.set_ylim(0, np.max(p)*2./pstar)

ax2 = ax1.twinx()
ax2.plot(r/Radius, area_fraction, "-", color="green", linewidth=2, label="Area fraction")
ax2.set_ylabel("Area fraction", color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(0, np.max(area_fraction)*1.1)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig(f"Final_pressure_ind_type_{ind_type}_approach_{d/sigma:.2f}.pdf")
plt.show()

# Final state plot
if ind_type == 'sphere':
    idx = np.searchsorted(r, 3*a_computed)
    plot_state(r[:idx], p[:idx], w0[:idx], w[:idx], Radius, sigma, pstar,
               fr"Converged Solution, $d/\sigma={d/sigma:.2f}$",
               f"Current_state_ind_type_{ind_type}_approach_{d/sigma:.2f}_iter.pdf", set_limits=True)
else:
    plot_state(r, p, w0, w, Radius, sigma, pstar,
               fr"Converged Solution, $d/\sigma={d/sigma:.2f}$",
               f"Current_state_ind_type_{ind_type}_approach_{d/sigma:.2f}_iter.pdf", set_limits=True)

# =============================================================================
#       CONTACT AREA VISUALIZATION
# =============================================================================

if ind_type == 'sphere':
    print("\nGenerating contact area visualization...")
    
    dx = np.sqrt(1/eta)
    extent = 2.0
    x_coarse = np.arange(-extent*a_computed, extent*a_computed+dx, dx)
    y_coarse = np.arange(-extent*a_computed, extent*a_computed+dx, dx)
    Xc, Yc = np.meshgrid(x_coarse, y_coarse, indexing='ij')
    Ac = np.zeros_like(Xc)
    number_of_asperities = eta * dx**2
    
    print(f"Number of asperities per coarse cell: {number_of_asperities:.2f}")
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlabel(r"Normalized X coordinate, $x/a$")
    ax.set_ylabel(r"Normalized Y coordinate, $y/a$")
    
    patches = []
    for i in range(Xc.shape[0]):
        for j in range(Xc.shape[1]):
            polar_r = np.sqrt(Xc[i,j]**2 + Yc[i,j]**2)
            area_frac = np.interp(polar_r, r, area_fraction, left=0.0, right=0.0)
            contact_area_total = area_frac * dx**2
            Ac[i,j] = contact_area_total
            contact_area_per_asperity = contact_area_total / number_of_asperities if number_of_asperities > 0 else 0
            
            if contact_area_per_asperity > 0:
                R = np.sqrt(contact_area_per_asperity / np.pi) / a_computed
                patches.append(CirclePolygon((Xc[i, j]/a_computed, Yc[i, j]/a_computed),
                                             radius=R, resolution=128, fc='grey', ec='none'))
    
    # Compute Holm's radius
    print("Computing Holm's radius...")
    holm_radius = compute_holm_radius(Xc, Yc, Ac)
    
    # Draw all circles efficiently
    pc = PatchCollection(patches, match_original=True)
    ax.add_collection(pc)
    
    # Legend handle
    ax.scatter([], [], s=30, color='grey', label='Asperity contact')
    
    # Reference circles
    ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False, linestyle='--',
                             label='Hertzian contact radius, $a$', zorder=3))
    ax.add_artist(plt.Circle((0, 0), holm_radius/a_computed, color='red', fill=False,
                             linestyle='--', label="Holm's radius", zorder=3))
    
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f"Contact_area_ind_type_{ind_type}_approach_{d/sigma:.2f}.pdf")
    plt.show()

print("\nSimulation complete!")