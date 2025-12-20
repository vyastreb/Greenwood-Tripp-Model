import rfgen as rf
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfcinv
from Greenwood_Tripp_Model import compute_holm_radius, F1_analytical

# LaTeX-style plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 8})

# Load results from Greenwood-Tripp model
try:
    data = np.load("Results_ind_type_sphere_full_history.npz")
    print("Loaded simulation results.")
    
    # Overwrite parameters with loaded values
    eta = float(data['eta'])
    beta = float(data['beta'])
    Radius = float(data['Radius'])
    sigma = float(data['sigma'])
    
except FileNotFoundError:
    print("Results file not found. Using default parameters.")
    # Roughness parameters
    eta = 2e8           # Asperity density (1/m²)
    beta = 0.3e-4       # Asperity tip radius (m)
    # Indenter geometry
    Radius = 0.01           # Indenter radius (m)
    sigma = 20e-6

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
Ac = np.zeros_like(Xc)
Ac_oxidized = np.zeros_like(Xc)

# Oxide correlation length
L = 20 / eta**0.5
Hurst = 0.5
kl = size / (2*L) 
ks = 3 * kl

# Oxide fraction
fract = 0.7

# Resistivity
resistivity = 2.7e-8  # Ohm meter (Al)

# Generate rough surface
N = 1024
Nsurfaces = 100
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

######################################
#  LOOP OVER APPROACHES
#####################################
Resistance_self = np.zeros_like(D)
Resistance_inter = np.zeros_like(D)

Oxidized_Resistance = np.zeros((H.shape[0], Nsurfaces))
for k in range(H.shape[0]):        
    area_fraction = np.pi * beta * eta * F1_analytical(H[k], sigma)
    sum_radii = 0.0
    for i in range(Xc.shape[0]):
        for j in range(Xc.shape[1]):
            polar_r = np.sqrt(Xc[i,j]**2 + Yc[i,j]**2)
            area_frac = np.interp(polar_r, r, area_fraction, left=0.0, right=0.0)
            contact_area_total = area_frac * dx**2
            Ac[i,j] = contact_area_total
            if contact_area_total > 0:
                sum_radii += np.sqrt(contact_area_total/np.pi)
    
    # Resistance calculation (assuming conductivity = 1 for now, user can adjust)
    # R_total = R_Holm + R_spots
    # R_spots = resistivity / (2 * sum(a_i))
    # R_Holm = resistivity / (2 * alpha)
    
    R_spots = 1.0 / (2.0 * sum_radii) if sum_radii > 0 else np.inf
    R_Holm = 1.0 / (2.0 * Holm_radius[k]) if Holm_radius[k] > 0 else np.inf
    
    Resistance_self[k] = R_spots
    Resistance_inter[k] = R_Holm

# Check with surface roughness
    # Vectorized oxide mapping
    # Resample oxide maps to match Xc shape
    # We can use simple block averaging or nearest neighbor if the grid aligns
    # Here we use nearest neighbor as in the original loop: int(i*N/Xc.shape[0])
    
    scale_x = N / Xc.shape[0]
    scale_y = N / Xc.shape[1]
    
    # Create indices for resampling
    idx_i = (np.arange(Xc.shape[0]) * scale_x).astype(int)
    idx_j = (np.arange(Xc.shape[1]) * scale_y).astype(int)
    
    # Use meshgrid to get 2D indices
    II, JJ = np.meshgrid(idx_i, idx_j, indexing='ij')
    
    for j in range(Nsurfaces):
        oxide = Oxide_maps[j]
        
        # Vectorized resampling of oxide map
        oxide_resampled = oxide[II, JJ]
        
        # Element-wise multiplication
        Ac_oxidized = Ac * (1 - oxide_resampled)
        
        # Vectorized sum of radii
        # Avoid sqrt of negative numbers (though Ac_oxidized should be >= 0)
        valid_mask = Ac_oxidized > 0
        sum_radii_oxide = np.sum(np.sqrt(Ac_oxidized[valid_mask] / np.pi))

        R_spots_oxide = 1.0 / (2.0 * sum_radii_oxide) if sum_radii_oxide > 0 else np.inf
        GreenwoodHolmRadius = compute_holm_radius(Xc, Yc, Ac_oxidized)
        R_Holm_oxide = 1.0 / (2.0 * GreenwoodHolmRadius) if GreenwoodHolmRadius > 0 else np.inf

        Oxidized_Resistance[k,j] = R_spots_oxide + R_Holm_oxide



fig,ax = plt.subplots(1,1, figsize=(3.,3))
plt.xscale("log")
plt.yscale("log")
plt.xlim(np.min(Force),np.max(Force))

for j in range(Nsurfaces):
    plt.plot(Force, resistivity * Oxidized_Resistance[:,j], "-", color="gray",alpha=0.1)

rms = np.std(resistivity * Oxidized_Resistance, axis=1)
mean = np.mean(resistivity * Oxidized_Resistance, axis=1)
plt.grid()
# plt.fill_between(Force, mean - rms, mean + rms, color='r', alpha=0.5)
plt.plot(Force, mean, "s-", color="yellowgreen", label=f"Mean oxidized, $f_{{\\mathcal{{O}}}}={fract}$, $l_{{\\mathcal{{O}}}}\\sqrt{{\\eta}}={L*np.sqrt(eta):.1f}$")

plt.plot(Force, resistivity * (Resistance_self + Resistance_inter), "o-", color="turquoise", label="No oxide")

# for fraction 0.7
plt.plot(Force, Force**(-0.33) * 0.47e-3, "--", color="k", label=r"$R \propto F^{-1/3}$")

# for fraction 0.5
# plt.plot(Force, Force**(-0.3) * 0.3e-3, "--", color="k", label=r"$R \propto F^{-0.3}$")

# for fraction 0.2
# plt.plot(Force, Force**(-0.25) * 0.165e-3, "--", color="k", label=r"$R \propto F^{-0.25}$")

plt.plot(Force, Force**(-0.23) * 1.33e-4, "-.", color="k", label=r"$R \propto F^{-0.23}$")


plt.xlabel("Force (N)")
plt.ylabel("Resistance (Ohm)")
plt.legend()
plt.show()
fig.savefig(f"Oxide_Conductivity_vs_Force_oxide_fraction_{fract}_correlation_{L*np.sqrt(eta)}.pdf", bbox_inches='tight', pad_inches=0, dpi=300)
