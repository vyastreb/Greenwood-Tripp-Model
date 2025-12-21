import rfgen as rf
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfcinv

# LaTeX-style plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 8})

# Roughness parameters
eta = 2e8           # Asperity density (1/m²)
beta = 0.3e-4       # Asperity tip radius (m)

# Indenter geometry
Radius = 0.01           # Indenter radius (m)

# Oxide correlation length
L = np.array([2, 5, 10]) / eta**0.5
Hurst = 0.5

# Oxide fraction
fract = np.array([0.5, 0.2, 0.1])

# Generate rough surface
size = 0.4 * Radius
N = 1024


rng = np.random.default_rng(42)

Surface = np.zeros((len(L), N, N))

fig,ax = plt.subplots(2, 3, figsize=(6.1,4.), sharex=True, sharey=True, gridspec_kw={'wspace': 0, 'hspace': 0.2})
for i,(l,fract) in enumerate(zip(L, fract)):
    kl = size / (2*l) #np.pi / l * size
    ks = 3 * kl
    Surface[i] = rf.selfaffine_field(
        dim=2, 
        N=N,
        Hurst=Hurst,
        k_low=kl/N,
        k_high=ks/N,
        plateau=False,
        noise=False,
        rng=rng)

    # Normalize to desired RMS
    current_rms = np.std(Surface[i])
    Surface[i] /= current_rms

    ax[0,i].imshow(Surface[i], cmap='coolwarm', extent=[-size/2./Radius, size/2./Radius, -size/2./Radius, size/2./Radius], vmin = -2, vmax = 2, interpolation = "bicubic")
    G0 = np.sqrt(2) * erfcinv(2*fract)
    oxide = Surface[i] > G0
    print("Error = np.sum(oxide)/oxide.size - fract =", abs(np.sum(oxide)/oxide.size - fract))
    ax[1,i].imshow(oxide, cmap='gray_r', extent=[-size/2./Radius, size/2./Radius, -size/2./Radius, size/2./Radius], interpolation = "none")
    ax[0,i].set_title(f"$l_{{\\mathcal{{O}}}}\sqrt{{\\eta}} = {l*np.sqrt(eta):.1f}$, $\\frac{{R}}{{l_{{\\mathcal{{O}}}}}} = {Radius/l:.1f}$")
    ax[1,i].set_title(f"$\\xi = {fract:.2f}$")
    
    # Only set labels for the appropriate axes
    if i == 0:
        ax[0,i].set_ylabel("$y/R$")
        ax[1,i].set_ylabel("$y/R$")
    
    ax[1,i].set_xlabel("$x/R$")

plt.tight_layout()
fig.savefig("Oxide_surfaces.pdf", bbox_inches='tight', pad_inches=0, dpi=300)
plt.show()
