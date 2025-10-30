"""
Greenwood-Tripp contact model [A] for rough surfaces
Implements the multi-asperity contact theory with iterative pressure-displacement coupling for arbitrary axi-symmetric indenters


**Remarks**

+ Eq. (4) in [A] seems to be incorrect. An appropriate equation from Johnson's book [B] is used instead (integral of Eq. (3.96a) in [B]).

**References:**

+ [A] Greenwood, J.A. and Tripp, J.H., The Elastic Contact of Rough Spheres, Journal of Applied Mechanics, 34(1), p.153-159 (1967).
+ [B] Johnson, K.L., Contact Mechanics, Cambridge University Press (1985). Ninth printing 2003.

Author: V.A. Yastrebov (Mines Paris - PSL, CNRS)
Date: September 2025
License: BSD 3-Clause License
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.special import ellipk
from numba import njit,prange
import matplotlib as mpl
from matplotlib.patches import CirclePolygon
from matplotlib.collections import PatchCollection

# Make LaTeX-style plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 10})

@njit(parallel=True,nopython=True)
def compute_holm_radius(Xc, Yc, Ac):
    numerator = 0
    denominator = 0
    for i in prange(Xc.shape[0]):
        for j in prange(Xc.shape[1]):
            denominator += Ac[i,j]
            for k in prange(Xc.shape[0]):
                for l in prange(Xc.shape[1]):
                    if not (i == k and j == l):
                        dist = np.sqrt((Xc[i,j] - Xc[k,l])**2 + (Yc[i,j] - Yc[k,l])**2)
                        numerator += Ac[i,j]*Ac[k,l] / dist

    alpha_inv = numerator / (denominator**2 * np.pi)
    return 1/alpha_inv/2.

def indenter_shape(r, type, params):
    """
    Define the shape of the indenter.
    Currently supports:
    - 'sphere': params = [radius]
    - 'cone': params = [half_angle_in_radians]
    - 'flat': params = [radius] (flat punch radius)
    """
    r = np.asarray(r)
    if type == 'sphere':
        R = params[0]
        return r**2 / (2 * R)
    elif type == 'cone':
        alpha = params[0]
        return r * np.tan(alpha)
    elif type == 'flat':
        R_flat = params[0]
        fillet_radius = params[1] if len(params) > 1 else 0.0
        if fillet_radius > 0:
            print(f"Flat punch radius: {R_flat} with fillet radius: {fillet_radius}")
            z = np.zeros_like(r)
            mask1 = r <= R_flat - fillet_radius
            z[mask1] = 0.0
            mask2 = (r > R_flat - fillet_radius) & (r < R_flat)
            z[mask2] = sigma*(fillet_radius - np.sqrt(fillet_radius**2 - (r[mask2] - R_flat+fillet_radius)**2))/fillet_radius
            # z[mask2] = sigma * (r[mask2] - R_flat+fillet_radius)/fillet_radius
            # print(np.max(z[mask2]), np.min(z[mask2]))
            mask3 = r >= R_flat
            z[mask3] = 1.
            return z
        else:
            print(f"Flat punch radius: {R_flat}")
            return np.where(r <= R_flat, 0.0, 1.)
    else:
        raise ValueError("Unsupported indenter type. Use 'sphere', 'cone', or 'flat'.")

def probability_density(x, sigma):
    """Gaussian probability density function for surface heights"""
    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-x**2/(2*sigma**2))
    
def F_scalar_32(u, sigma):
    """
    Compute F_{3/2} integral for Greenwood-Tripp model
    Integral from u to infinity of (z-u)^{3/2} * phi(z) dz
    """
    # For numerical stability, return 0 if far from contact region
    if u > 20*sigma:
        return 0.0
    
    # Integrate over reasonable range where integrand is non-negligible
    integrand = lambda z: (z - u)**1.5 * probability_density(z, sigma)
    result, _ = integrate.quad(integrand, u, u + 10*sigma, epsabs=1e-8, epsrel=1e-8)
    return result

def F_scalar_1(u, sigma):
    """
    Compute F_{1} integral for Greenwood-Tripp model
    Integral from u to infinity of (z-u)^{1} * phi(z) dz
    """
    # For numerical stability, return 0 if far from contact region
    if u > 20*sigma:
        return 0.0
    
    # Integrate over reasonable range where integrand is non-negligible
    integrand = lambda z: (z - u) * probability_density(z, sigma)
    result, _ = integrate.quad(integrand, u, u + 10*sigma, epsabs=1e-8, epsrel=1e-8)
    return result
def F32(h, sigma):
    """Vectorized version of F_{3/2} integral"""
    if np.isscalar(h):
        return F_scalar_32(h, sigma)
    return np.array([F_scalar_32(hi, sigma) for hi in np.atleast_1d(h)])

def F1(h, sigma):
    """Vectorized version of F_{1} integral"""
    if np.isscalar(h):
        return F_scalar_1(h, sigma)
    return np.array([F_scalar_1(hi, sigma) for hi in np.atleast_1d(h)])

def integrand(x, rv, p_interpol):
    """
    Integrand for computing displacement from pressure using Boussinesq solution
    Accounts for circular contact geometry with elliptic integral
    See Johnson [B], Eq. (3.96a)
    """
    eps = 1e-10  # Small value to avoid division by zero
    k = 4 * x * rv / (rv + x + eps)**2  # Parameter for elliptic integral
    return x / (x + rv + eps) * p_interpol(x) * ellipk(k)
    # Note in Johnson [B], Eq. (3.96a), the convention is that K(k) is the complete elliptic integral of the first kind, which corresponds to ellipk(k^2) in scipy

def get_displacement_from_p(p_vals, r_grid, Estar, integration_limit):
    """
    Compute surface displacement from pressure distribution
    Uses Boussinesq solution for elastic half-space
    """
    r = np.asarray(r_grid)
    coef = 4 / (np.pi * Estar)  # Elastic coefficient
    result = np.zeros_like(r, dtype=float)
    
    # Create interpolation function for pressure
    p_interp = lambda x: np.interp(x, r, p_vals, left=0.0, right=0.0)

    # Integrate for each radial position
    for i, rv in enumerate(r):
        val, _ = integrate.quad(
            lambda x: integrand(x, rv, p_interp), 
            1e-9, integration_limit, 
            epsabs=1e-10, epsrel=1e-8
        )
        result[i] = coef * val
    
    return result if r.ndim > 0 else result[0]

def plot_current_state(r, p, z, w, title="Current state"):
    """
    Visualization function to monitor convergence progress
    Shows pressure distribution, displacement, and displacement change
    """
    fig, ax = plt.subplots(1, 3, figsize=(5.1, 1.6))
    fig.suptitle(title)

    # Pressure distribution
    ax[0].grid()
    ax[0].set_xlim(0, np.max(r/Radius))
    ax[0].set_ylim(0, np.max(p/pstar)*1.1)
    ax[0].plot(r/Radius, p/pstar, "-", label="Pressure")
    ax[0].legend()
    ax[0].set_ylabel(r"Normalized pressure, $p/\bar p$")
    ax[0].set_xlabel("Normalized radial coordinate, $r/R$")

    # Surface displacement with reference lines
    ax[1].grid()
    ax[1].set_xlim(0, np.max(r/Radius))
    ylim = min(z[0]/sigma*1.1, -0.1)
    print("ylim for displacement plot:", ylim, ", z[0]=", z[0])
    ax[1].set_ylim(ylim, 7)
    ax[1].plot(r/Radius, z/sigma, "--", color = "k", label="Reference shape")
    ax[1].plot(r/Radius, (z+w)/sigma, "-", color="#1f77b4", label="Deformed shape", zorder=3)
    ax[1].axhline(0, color='grey', linestyle='--', alpha=1, label="Mean surface")
    ax[1].axhline(1, color='grey', linestyle='--', alpha=0.5, label="$\\sigma$")
    ax[1].axhline(2, color='grey', linestyle='--', alpha=0.25, label="2$\\sigma$")
    ax[1].axhline(3, color='grey', linestyle='--', alpha=0.125, label="3$\\sigma$")
    ax[1].set_ylabel(r"Normalized configuration, $z/\sigma$")
    ax[1].set_xlabel("Normalized radial coordinate, $r/R$")
    ax[1].legend()

    # Displacement change for convergence monitoring
    ax[2].grid()
    ax[2].set_xlim(0, np.max(r/Radius))
    ax[2].plot(r/Radius, w/sigma, label="Displacement")
    ax[2].set_ylabel(r"Normalized displacement, $u/\sigma$")
    ax[2].set_xlabel("Normalized radial coordinate, $r/R$")
    ax[2].legend()
    
    # plt.show()
    fig.savefig(f"Current_state_ind_type_{ind_type}_approach_{d/sigma:.2f}_iter.pdf")


# =============================================================================
# MATERIAL AND GEOMETRIC PARAMETERS
# =============================================================================

# Macroscopic contact properties
E = 2.1e11                   # Young's modulus (Pa)
nu = 0.3                  # Poisson's ratio
Estar = E / (2*(1 - nu**2))  # Reduced elastic modulus (assume same material)

# Surface roughness parameters
sigma = 1e-6              # RMS surface roughness (m)
eta = 30e3 * 1e6           # Asperity density (1/m²)
# eta = 2 * 1e6             # Asperity density (1/m²)
beta = 1e-4               # Asperity tip radius of curvature (m)
mu = (4.0/3.0) * eta * Estar * np.sqrt(beta)  # Contact stiffness parameter
chi = np.pi * eta * beta # Asperity area parameter

# Indenter geometry
ind_type = 'sphere'        # Indenter type: 'sphere', 'cone', or 'flat'
Radius = .01               # Indenter radius (m)
# ind_type = "flat"
# Radius = .5              # Indenter radius (m)
# fillet_radius = 0.05     # Fillet radius for flat punch (m)


# Numerical parameters
d = -2 * sigma            # Initial surface separation (m)
if ind_type == 'sphere':
    params = [Radius]
    integration_limit = 0.1 * Radius  # Radial integration limit (m)
elif ind_type == 'flat':
    params = [Radius, fillet_radius]
    integration_limit = 2. * Radius
elif ind_type == 'cone':
    integration_limit = 0.1 * Radius
pstar = Estar * np.sqrt(sigma / Radius)  # Characteristic pressure scale
Np = 30                 # Number of radial grid points

# Convergence parameters
kappa = 0.2               # Under-relaxation factor for stability
tolerance = 1e-3          # Convergence tolerance
max_iter = 100            # Maximum number of iterations

# =============================================================================
# INITIALIZATION
# =============================================================================

# Create radial grid
r = np.linspace(0, integration_limit, Np)

# Initialize displacement arrays
w = np.zeros_like(r)
w_old = np.zeros_like(r)

# Initial guess: rigid body displacement plus sphere geometry
w0 = d + indenter_shape(r, type=ind_type, params=params)
w = w0.copy()

# Initial pressure distribution from Greenwood-Tripp model
p = mu * F32(w, sigma)
print(f"Initial pressure computed, max value: {np.max(p):.2e} Pa")

# Get initial displacement from pressure distribution
w_new = get_displacement_from_p(p, r, Estar, integration_limit)


# Show initial state
plot_current_state(r, p, w0, w_new, "Initial state")


# =============================================================================
# ITERATIVE SOLUTION
# =============================================================================

# Surface height including initial separation and sphere geometry
h = d + indenter_shape(r, ind_type, params)

print("Starting iterative solution...")

# GRADUAL_APPROACH = True  # Gradually decrease separation to improve convergence
# if GRADUAL_APPROACH:
#     d_values = np.linspace(3*sigma, d, 10)
#     for inc,d in enumerate(d_values):
#         h = d + indenter_shape(r, ind_type, params=[Radius])
#         print(f"Approach step: d/sigma = {d/sigma:.2f}")
#         for it in range(max_iter):  # Few iterations per approach step
#             # Compute new displacement from current pressure
#             w_new = get_displacement_from_p(p, r, Estar, integration_limit)
            
#             # Apply under-relaxation for stability
#             w = kappa * w_new + (1 - kappa) * w_old
            
#             # Update surface height accounting for displacement change
#             h += (w - w_old)
            
#             # Compute new pressure from updated surface height
#             p = mu * F32(h, sigma)
            
#             # Check convergence
#             eps = np.max(np.abs(w - w_old)/kappa) / (np.max(np.abs(w_old)) + 1e-20)
#             w_old = w.copy()

#             # Print iteration info (erase previous iteration and print new)
#             print(f"\r--> Iteration {it + 1:03d}/{max_iter}, Tolerance: {eps:.2e}", end='', flush=True)

#             if eps < tolerance:
#                 print(f"Converged in {it} iterations (error: {eps:.2e})")
#                 break
            
#         # Show current state after each approach step
#         plot_current_state(r, p, w0, w, title=f"State at d/sigma={d/sigma:.2f}")
# else:
eps_old = 1e10
for it in range(max_iter):
    # Compute new displacement from current pressure
    w_new = get_displacement_from_p(p, r, Estar, integration_limit)
    
    # Define kappa
    # kappa = max(np.min(h) / np.max(w_new), 0.01)

    # Apply under-relaxation for stability
    w = kappa * w_new + (1 - kappa) * w_old
    
    # Update surface height accounting for displacement change
    h += (w - w_old)
    
    # Compute new pressure from updated surface height
    p = mu * F32(h, sigma)
    
    # Check convergence
    eps = np.max(np.abs(w - w_old)/kappa) / (np.max(np.abs(w_old)) + 1e-20)
    w_old = w.copy()
    if it > 2 and eps > eps_old*1.01:
        kappa = max(kappa * 0.5, 0.01)
        print(f"\nReducing kappa to {kappa:.3f} due to increase in error")
    eps_old = eps

    # Print iteration info (erase previous iteration and print new)
    print(f"\r--> Iteration {it + 1:03d}/{max_iter}, Tolerance: {eps:.2e}", end='', flush=True)

    if eps < tolerance:
        print(f"Converged in {it} iterations (error: {eps:.2e})")
        break
    
    if it == max_iter - 1:
        print(f"Warning: did not converge after {max_iter} iterations (error: {eps:.2e})")
            
# =============================================================================
# RESULTS AND COMPARISON WITH HERTZIAN SOLUTION
# =============================================================================

# Compute total force by integrating pressure in cylindrical coordinates
force_computed = 2 * np.pi * integrate.simps(p * r, r)
print(f"Computed force: {force_computed:.3f} N")

if ind_type == 'sphere':
    # Compute equivalent Hertzian contact parameters
    a_computed = ((3 * force_computed * Radius) / (4 * Estar))**(1/3.)
    delta_computed = a_computed**2 / Radius
    p0 = (3 * force_computed) / (2 * np.pi * a_computed**2)
    ref_r = np.linspace(0, a_computed, 100)
    ref_pressure = p0 * np.sqrt(1 - (ref_r / a_computed) ** 2)
    print(f"Roughness parameter sigma*R/a^2: {sigma*Radius/(a_computed**2):.3f}")

elif ind_type == 'flat':
    p0 = force_computed / (np.pi * Radius**2)
    ref_r = np.linspace(0, Radius*(1-1e-4), 100)
    ref_pressure = 0.5 * p0 / np.sqrt(1 - ref_r**2 / Radius**2)

area_fraction = chi * F1(h, sigma)

# Plot pressure comparison with area fraction
fig, ax1 = plt.subplots(figsize=(5.1, 3.))

# Plot pressure on primary y-axis
ax1.plot(ref_r/Radius, ref_pressure/pstar, "r--", linewidth=2, label="Hertzian pressure")
ax1.plot(r/Radius, p/pstar, "-", color="#1f77b4", linewidth=2, label="Greenwood-Tripp pressure")

if ind_type == 'sphere':
    ax1.set_xlim(0, 3*a_computed/Radius)
    ax1.set_ylim(0, np.max(ref_pressure)*1.1/pstar)
    # ax1.set_ylim(0, np.max(p)*1.1/pstar)
else:
    ax1.set_xlim(0, np.max(r/Radius))
    ax1.set_ylim(0, np.max(p)*2./pstar)

ax1.set_xlabel("Normalized radial coordinate, $r/R$")
ax1.set_ylabel(r"Normalized pressure, $p/\bar p$", color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best')

# Create secondary y-axis for area fraction
ax2 = ax1.twinx()
ax2.plot(r/Radius, area_fraction, "-", color="green", linewidth=2, label="Area fraction")
ax2.set_ylabel("Area fraction", color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(0, np.max(area_fraction)*1.1)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

ax1.set_title("Pressure Distribution and Area Fraction Comparison")
plt.tight_layout()
plt.show()
fig.savefig(f"Final_pressure_ind_type_{ind_type}_approach_{d/sigma:.2f}.pdf")

# Show final converged state
# Determine index of r becoming > 3*a_computed
if ind_type == 'sphere':
    idx = r.shape[0] 
    idx = np.searchsorted(r, 3*a_computed)
    plot_current_state(r[:idx], p[:idx], w0[:idx], w[:idx], title=fr"Converged Solution, $d/\sigma={d/sigma:.2f}$")
else:
    plot_current_state(r, p, w0, w, title=fr"Converged Solution, $d/\sigma={d/sigma:.2f}$")


# Show contact area
dx = np.sqrt(1/eta)
extent = 2.
x_coarse = np.arange(-extent*a_computed, extent*a_computed+dx, dx)
y_coarse = np.arange(-extent*a_computed, extent*a_computed+dx, dx)
Xc, Yc = np.meshgrid(x_coarse, y_coarse, indexing='ij')
Ac = np.zeros_like(Xc)
Rc = np.sqrt(Xc**2 + Yc**2)
number_of_asperities = eta * dx**2
print(f"Number of asperities per coarse cell: {number_of_asperities:.2f}")
fig,ax = plt.subplots(figsize=(6, 6))
plt.xlabel("Normalized X coordinate, $x/a$")
plt.ylabel("Normalized Y coordinate, $y/a$")
first_contact = True  # Flag to add label only once
patches = []
for i in range(Xc.shape[0]):
    for j in range(Xc.shape[1]):
        polar_r = np.sqrt(Xc[i,j]**2 + Yc[i,j]**2)
        # Exact computation of area fraction at polar_r
        area_frac = np.interp(polar_r, r, area_fraction, left=0.0, right=0.0)
        # Total contact area in the cell (dx x dx) containing number_of_asperities asperities
        contact_area_total = area_frac * dx**2
        Ac[i,j] = contact_area_total
        # Average contact area per asperity in this cell
        contact_area_per_asperity = contact_area_total / number_of_asperities if number_of_asperities > 0 else 0
        
        # Only plot circles if there's contact
        if contact_area_per_asperity > 0:
            label = 'Asperity contact' if first_contact else ''
            R = np.sqrt(contact_area_per_asperity / np.pi) / a_computed
            # Circle with high vertex resolution (stays circular even when tiny)
            patches.append(CirclePolygon((Xc[i, j]/a_computed, Yc[i, j]/a_computed),
                                         radius=R, resolution=128, fc='grey', ec='none'))
            # circle = plt.Circle((Xc[i,j]/a_computed, Yc[i,j]/a_computed), np.sqrt(contact_area_per_asperity/np.pi)/a_computed, 
            #                   color='grey', fill=True, alpha=1., label=label)
            # ax.add_artist(circle)
            first_contact = False

# Compute Holm's radius
print("Computing Holm's radius...")
holm_radius = compute_holm_radius(Xc, Yc, Ac)

# draw all circles efficiently at once
pc = PatchCollection(patches, match_original=True)
ax.add_collection(pc)

# invisible scatter to keep legend handle spacing if you need it
plt.scatter([], [], s=30, color='grey', label='Asperity contact')

# reference circles
ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False, linestyle='--',
                         label='Hertzian contact radius, $a$', zorder=3))

ax.add_artist(plt.Circle((0, 0), holm_radius/a_computed, color='red', fill=False,
                         linestyle='--', label="Holm's radius", zorder=3))

plt.legend()
plt.show()
fig.savefig(f"Contact_area_ind_type_{ind_type}_approach_{d/sigma:.2f}.pdf")


# =============================================================================
#   PRINT RESULTS IN FILE
# =============================================================================

if ind_type == 'sphere':
    max_penetration = np.min(w0 + w)
    max_pressure = np.max(p)
    hertz_contact_radius = a_computed
    max_hertz_pressure = p0
    roughness_parameter = sigma*Radius/(a_computed**2)

    filename = "Results_Greenwood_Tripp_sphere.data"
    with open(filename, 'a') as f:
        f.write(f"{sigma:.3e} {d/sigma:.3f} {max_penetration/sigma:.3f} {force_computed:.3f} {hertz_contact_radius:.6f} {max_pressure:.1f} {max_hertz_pressure:.1f} {roughness_parameter:.3f}\n")
    print(f"Results written to {filename}")

