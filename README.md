# Greenwood-Tripp Rough Contact Axisymmetric Model

## Description

This is a Python implementation of Greenwood-Tripp rough elastic contact model in which both averaged elastic deformation and rough contact are taken into account. 

Note that Eq. in [1] seems to be incorrect. An appropriate equation from Johnson's book [2] is used instead (integral of Eq. (3.96a) in [2]).

+ [1] Greenwood, J.A. and Tripp, J.H., The Elastic Contact of Rough Spheres, Journal of Applied Mechanics, 34(1), p.153-159 (1967).
+ [2] Johnson, K.L., Contact Mechanics, Cambridge University Press (1985). Ninth printing 2003.

## Input parameters

Parameters have to be provided in the code.

+ Macroscopic contact properties
    + `E  = 2.1e11               # Young's modulus (Pa)`
    + `nu = 0.3                  # Poisson's ratio`
+ Surface roughness parameters
    + `sigma = 1e-6              # RMS surface roughness (m)`
    + `eta = 4e3 * 1e6           # Asperity density (1/m^2)`
    + `beta = 1e-5               # Asperity tip radius of curvature (m)`
+ Indenter geometry
    + `ind_type = 'sphere'        # Indenter type: 'sphere', 'cone', or 'flat'`
    + `Radius = .01               # Indenter radius (m)`
+ Numerical parameters
    + `d = -2 * sigma            # Initial surface separation (m)`
    + `Np = 50                 # Number of radial grid points`
+ Convergence parameters
    + `kappa = 0.2               # Under-relaxation factor for stability`
    + `tolerance = 1e-3          # Convergence tolerance`
    + `max_iter = 100            # Maximum number of iterations`

## Information

+ **Author:** Vladislav A. Yastrebov (CNRS, Mines Paris - PSL)
+ **License:** BSD-3 clause
+ **Date:** Nov-Dec 2025
+ **AI-usage:** Copilot in VScode with different models was used for simple adjustments