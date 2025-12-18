---
title: "Contact of Rough Spheres: Greenwood-Tripp model"
author: Vladislav A. Yastrebov
date: CNRS, Mines Paris - PSL, Versailles, France
header-includes:
  - \AtBeginDocument{\hypersetup{colorlinks=true,linkcolor=navy,citecolor=navy,urlcolor=navy}}
  - \usepackage{siunitx}
  - \usepackage{amsmath}
  - \usepackage{graphicx}
  - \usepackage{xcolor}
  - \definecolor{navy}{RGB}{0,0,128}
  - \usepackage{hyperref}
  - \usepackage{cleveref}
---

## Model description

To make a truly predictive model linking contact area and force in contact of rough spheres, Greenwood and Tripp constructed a model taking into account averaged deformations induced by the contact pressure due to asperity contact [1]. This model, initially constructed for spherical contact, could be extended to arbitrary contact shape (its implementation for the axisymmetric contact is provided in \href{https://github.com/vyastreb/Greenwood-Tripp-Model}{github.com/vyastreb/Greenwood-Tripp-Model}).

The main feature of the model is that it solves the problem in an iterative way by computing pressures from the separation field using multi-asperity model, then it computes induced displacements using integral convolution, recomputing separations and so on before the convergence. The pressure $p^i(x,y)$ induced by asperity contact based on a statistical multi-asperity model for the separation $z^i_0(x,y)$ between two surfaces is given by the following equation
\begin{equation}
 p^i(x,y) = \tfrac{4}{3} \eta E^* r^{1/2}  \int\limits_{z^i_0(x,y)}^\infty (z-z^i_0(x,y))^{3/2}P(z) \,dz,
 \label{eq:gt1}
\end{equation}
where $\eta$ is the asperity density (\SI{}{m^{-2}}), $r$ (\SI{}{m}) their average geometrically mean curvature radius and $P(z)$ (\SI{}{m^{-1}}) is the probability density of asperity heights.
This average pressure is used to induce macroscopic deformation following Boussinesq approach, the total induced displacement (summing both sides) is given by
\begin{equation}
u^i(x,y) = \frac{1}{\pi E^*}\;\text{p.v.}\int\limits_{\mathbb R^2} \frac{p^i(x',y')\,dx'dy'}{\sqrt{(x-x')^2+(y-y')^2}},
 \label{eq:gt2}
\end{equation}
where p.v. denotes principal value of the integral. Then the separation between the surfaces is updated $z^{i+1}_0 = z^i_0 + u$, new pressure $p^{i+1}$ is recomputed \eqref{eq:gt1} as well as the induced displacement $z^{i+1}$ from \eqref{eq:gt2}.
For an axisymmetric problem, a simpler form is available [2][see Eq. (3.96a)]:
\begin{equation}
    u_z(r) = \frac{4}{\pi E^*} \int_0^\infty \frac{\rho}{\rho + r} \, p(\rho) \, K(k(r,\rho)) \, d\rho
\end{equation}
where $r$ and $\rho$ are the radial coordinate and the modulus $k$ of the complete elliptic integral of the first kind $K(k)$ is given by:
\begin{equation}
k(r,\rho) = \frac{4\rho r}{(r + \rho)^2}, \quad K(k) = \int_0^{\pi/2}[1-k\sin^2(t)]^{-1/2}\,dt.
\end{equation}
At the convergence separation $z^*(x,y)$, we have the information about contact area fraction, which can be computed as 
\begin{equation}
    A^*(x,y) = \pi \eta r \int\limits_{z^*}^\infty (z^*-z) P(z)\,dz.
\end{equation}

## Example

For the set of parameters provided in \Cref{tab:1}, the roughness parameter [3] can be evaluated
$$
\lambda = \frac{\sigma R}{a^2} \approx 0.361,
$$
placing this problem in the class of problems where the deviation from the Hertz theory is significant.
The iterative algorithm converges in 13 iterations for the tolerance of $\epsilon=10^{-3}$ defined by
$$
\frac{\|u_{k+1}-u_k\|_{\infty}}{\|u_k\|_{\infty}+10^{-20}} \le \epsilon.
$$
We use some relaxation technique by weighting the obtained displacement as
$$
 u_{k+1} = \kappa u_{k+1} + (1-\kappa) u_{k}.
$$
The obtained pressure and contact area distribution are shown in \Cref{fig:1}.
In \Cref{fig:2} the resulting pressure, deformed configuration and the resulting surface displacement are shown.


\begin{table}[h]
\centering
\begin{tabular}{lcccp{5cm}}
\hline
Parameter & Symbol & Value & Units & Description \\
\hline
\multicolumn{5}{l}{\textbf{Material Properties}} \\
Young's modulus & $E$ & $2.1 \times 10^{11}$ & Pa & Steel elastic modulus \\
Poisson's ratio & $\nu$ & $0.3$ & - & Poisson's ratio \\
Combined modulus & $E^*$ & $\approx 1.15 \times 10^{11}$ & Pa & $E/(2(1-\nu^2))$ \\
\hline
\multicolumn{5}{l}{\textbf{Roughness Parameters}} \\
RMS roughness & $\sigma$ & $20$ & \si{\micro\meter} & Root mean square height \\
Asperity density & $\eta$ & $2 \times 10^{8}$ & \si{\per\square\meter} & Number of asperities per unit area \\
Asperity tip radius & $\beta$ & $30$ & \si{\micro\meter} & Mean radius of curvature \\
\hline
\multicolumn{5}{l}{\textbf{Indenter Geometry}} \\
Type & - & Sphere & - & Indenter shape \\
Radius & $R$ & $10$ & \si{\milli\meter} & Indenter radius \\
\hline
\multicolumn{5}{l}{\textbf{Loading Parameters}} \\
Initial separation & $d$ & $-40$ & \si{\micro\meter} & $-2\sigma$ \\
\hline
\multicolumn{5}{l}{\textbf{Numerical Parameters}} \\
Convergence tolerance & $\epsilon$ & $10^{-3}$ & (-) & \\
Relaxation parameter & $\kappa$ & 0.2 & (-) & \\
\hline
\end{tabular}
\caption{Model parameters}
\label{tab:1}
\end{table}

\begin{figure}
\includegraphics[width=1\textwidth]{Final_pressure_ind_type_sphere_approach_-2.00.pdf}
\caption{\label{fig:1}Pressure and true contact area distribution for rough contact compared with the Hertz solution.}
\end{figure}

\begin{figure}
\includegraphics[width=1\textwidth]{Current_state_ind_type_sphere_approach_-2.00_iter.pdf}
\caption{\label{fig:2}Converged pressure distribution (first column), initial penetration (dashed line) and the resulting indenter's configuration (middle column), converged vertical displacement.}
\end{figure}

## References


\begin{enumerate}
\item J. A. Greenwood and J. H. Tripp (1967).
\textit{The elastic contact of rough spheres}.
Journal of Applied Mechanics, \textbf{34}(1), 153--159.
\href{https://doi.org/10.1115/1.3607616}{doi:10.1115/1.3607616}

\item K. L. Johnson (1985).
\textit{Contact Mechanics}.
Cambridge University Press, Cambridge.
Ninth printing, 2003.

\item J. A. Greenwood, K. L. Johnson, and E. Matsubara (1984).
\textit{A surface roughness parameter in Hertz contact}.
Wear, \textbf{100}(1--3), 47--57.
\href{https://doi.org/10.1016/0043-1648(84)90005-X}{doi:10.1016/0043-1648(84)90005-X}
\end{enumerate}
