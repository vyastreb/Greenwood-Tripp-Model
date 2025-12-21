import numpy as np
import matplotlib.pyplot as plt
import glob

# LaTeX-style plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 8})

# Parameters
L_indices = [0, 1, 2]
Fract_indices = [0, 1, 2]
colors = ['yellowgreen', 'orange', 'purple']
markers = ['s-', 'd-', '^-']

prefix = "Percentile_"

# Initialize fitting constants file
with open("fitting_constants.csv", "w") as f:
    f.write("oxide_fraction, correlation_sqrt_eta, a, b\n")

# 1. Individual plots for Resistance
print("Generating Resistance plots...")
for L_idx in L_indices:
    for fract_idx in Fract_indices:
        filename = f"Oxide_Results_L_{L_idx}_fract_{fract_idx}.npz"
        try:
            data = np.load(filename)
        except FileNotFoundError:
            print(f"File {filename} not found. Skipping.")
            continue
            
        Oxidized_Resistance = data['Oxidized_Resistance']
        Resistance_self = data['Resistance_self']
        Resistance_inter = data['Resistance_inter']
        Force = data['Force']
        L = float(data['L'])
        fract = float(data['fract'])
        eta = float(data['eta'])
        resistivity = float(data['resistivity'])
        Nsurfaces = Oxidized_Resistance.shape[1]

        fig, ax = plt.subplots(1, 1, figsize=(3., 3))
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(np.min(Force), np.max(Force))
        plt.ylim(1e-5, 2e-3)

        for j in range(Nsurfaces):
            plt.plot(Force, resistivity * Oxidized_Resistance[:, j], "-", color="gray", alpha=0.1, linewidth=0.5)

        # rms = np.std(resistivity * Oxidized_Resistance, axis=1)
        mean = np.mean(resistivity * Oxidized_Resistance, axis=1)
        p16 = np.percentile(resistivity * Oxidized_Resistance, 16, axis=1)
        p84 = np.percentile(resistivity * Oxidized_Resistance, 84, axis=1)
        plt.grid()
        
        color = colors[fract_idx]
        marker = markers[fract_idx]
        
        plt.fill_between(Force, p16, p84, color=color, alpha=0.2)
        plt.plot(Force, mean, marker, color=color, markersize=4.5, markeredgecolor='k', markeredgewidth=0.5,
                 label=f"Mean oxidized, $\\xi={fract}$, $l_{{\\mathcal{{O}}}}\\sqrt{{\\eta}}={L * np.sqrt(eta):.1f}$")

        plt.plot(Force, resistivity * (Resistance_self + Resistance_inter), "o-", color="grey", markerfacecolor='w', markersize=4.5, markeredgecolor='k', markeredgewidth=0.5, label="No oxide")

        # Power laws fitting
        force_min_fit = 200.0
        mask_fit = Force >= force_min_fit
        
        if np.sum(mask_fit) > 1:
            F_fit = Force[mask_fit]
            
            # Fit Clean Resistance
            R_clean = resistivity * (Resistance_self + Resistance_inter)
            R_clean_fit_data = R_clean[mask_fit]
            
            coeffs_clean = np.polyfit(np.log(F_fit), np.log(R_clean_fit_data), 1)
            b_clean = -coeffs_clean[0]
            a_clean = np.exp(coeffs_clean[1])
            
            plt.plot(F_fit, a_clean * F_fit**(-b_clean), "--", color="k", linewidth=0.75, label=rf"Fit clean: $F^{{-{b_clean:.2f}}}$")

            # Fit Oxidized Mean Resistance
            R_ox_mean = mean
            R_ox_fit_data = R_ox_mean[mask_fit]
            
            coeffs_ox = np.polyfit(np.log(F_fit), np.log(R_ox_fit_data), 1)
            b_ox = -coeffs_ox[0]
            a_ox = np.exp(coeffs_ox[1])
            
            plt.plot(F_fit, a_ox * F_fit**(-b_ox), "-.", color="k", linewidth=0.75, label=rf"Fit oxidized: $F^{{-{b_ox:.2f}}}$")

            with open("fitting_constants.csv", "a") as f:
                f.write(f"{fract}, {L * np.sqrt(eta)}, {a_ox}, {b_ox}\n")

        # plt.plot(Force, Force**(-0.23) * 1.33e-4, "-.", color="k", label=r"$R \propto F^{-0.23}$")

        plt.xlabel("Force (N)")
        plt.ylabel("Resistance (Ohm)")
        plt.legend()
        plot_filename = f"{prefix}Oxide_Conductivity_vs_Force_oxide_fraction_{fract}_correlation_{L * np.sqrt(eta):.1f}.pdf"
        fig.savefig(plot_filename, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close(fig)
        print(f"Saved {plot_filename}")

# 2. Combined plots for Conductivity
print("Generating Conductivity plots...")
for L_idx in L_indices:
    fig, ax = plt.subplots(1, 1, figsize=(3., 3))
    # Load one file to get common data
    first_fract_idx = Fract_indices[0]
    filename = f"Oxide_Results_L_{L_idx}_fract_{first_fract_idx}.npz"
    try:
        data = np.load(filename)
    except FileNotFoundError:
        print(f"File {filename} not found. Skipping L={L_idx}.")
        plt.close(fig)
        continue

    Resistance_self = data['Resistance_self']
    Resistance_inter = data['Resistance_inter']
    Force = data['Force']
    Radius = float(data['Radius'])
    sigma = float(data['sigma'])
    Estar = float(data['Estar'])
    L = float(data['L'])
    eta = float(data['eta'])
    
    Rstar = Radius
    s = sigma
    F_norm = 2 * Force / (Estar * np.sqrt(Rstar * s**3))
    K_norm = 2 / ((Resistance_self + Resistance_inter) * np.sqrt(Rstar * s))
    plt.xlim(0,np.max(F_norm))
    
    plt.plot(F_norm, K_norm, "o-", color="grey", markerfacecolor='w', markersize=4.5, markeredgecolor='k', markeredgewidth=0.5, label="No oxide")
    
    # Fit Clean Conductance
    force_min_fit = 200.0
    mask_fit = Force >= force_min_fit
    if np.sum(mask_fit) > 1:
        F_fit_norm = F_norm[mask_fit]
        K_clean_fit = K_norm[mask_fit]
        
        coeffs_clean = np.polyfit(np.log(F_fit_norm), np.log(K_clean_fit), 1)
        b_clean = coeffs_clean[0]
        a_clean = np.exp(coeffs_clean[1])
        
        plt.plot(F_fit_norm, a_clean * F_fit_norm**b_clean, "--", color="k", linewidth=0.75, label="Power law fits")
    
    for i, fract_idx in enumerate(Fract_indices):
        filename = f"Oxide_Results_L_{L_idx}_fract_{fract_idx}.npz"
        try:
            data = np.load(filename)
        except FileNotFoundError:
            continue
            
        Oxidized_Resistance = data['Oxidized_Resistance']
        fract = float(data['fract'])
        
        # rms = np.std(2 / (Oxidized_Resistance * np.sqrt(Rstar * s)), axis=1)
        data_norm = 2 / (Oxidized_Resistance * np.sqrt(Rstar * s))
        mean = np.mean(data_norm, axis=1)
        p16 = np.percentile(data_norm, 16, axis=1)
        p84 = np.percentile(data_norm, 84, axis=1)
        
        color = colors[i]
        marker = markers[i]
        
        plt.fill_between(F_norm, p16, p84, color=color, alpha=0.2)
        plt.plot(F_norm, mean, marker, color=color, markersize=4.5, markeredgecolor='k', markeredgewidth=0.5, label=f"$\\xi={fract}$")

        # Fit Oxidized Conductance
        if np.sum(mask_fit) > 1:
            F_fit_norm = F_norm[mask_fit]
            K_ox_fit = mean[mask_fit]
            
            coeffs_ox = np.polyfit(np.log(F_fit_norm), np.log(K_ox_fit), 1)
            b_ox = coeffs_ox[0]
            a_ox = np.exp(coeffs_ox[1])
            
            plt.plot(F_fit_norm, a_ox * F_fit_norm**b_ox, "--", color="k", linewidth=0.75)

    plt.xlabel(r"Normalized Force, $F'$")
    plt.ylabel(r"Normalized Conductance, $\kappa'$")
    plt.title(f"$l_{{\\mathcal{{O}}}}\\sqrt{{\\eta}}={L * np.sqrt(eta):.1f}$")
    plt.legend()
    
    plot_filename = f"{prefix}Oxide_Conductivity_Normalized_L_{L * np.sqrt(eta):.1f}.pdf"
    fig.savefig(plot_filename, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)
    print(f"Saved {plot_filename}")
