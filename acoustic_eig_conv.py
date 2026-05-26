'''
Filename: acoustic_eigs2.py
Author: Adam Kautzky
Date 2026-05-07
Description: This code performs eigenvalue error analysis for the spatial operator of the 2D acoustic wave eq, comparing a staggered-grid discretization and the SBP-projection method.
             The code creates a Tikz figure of the convergence plot.
Dependencies:
    numpy 1.26.4
    scipy 1.17.0
    matplotlib 3.7.5
'''




"""
Eigenvalue Convergence Comparison for Two Acoustic Wave Discretizations
=========================================================================

Purpose
-------
This script compares the eigenvalue accuracy of two finite-difference
discretizations of the two-dimensional first-order acoustic wave equation:

1. A second-order summation-by-parts (SBP) discretization with homogeneous
   pressure boundary conditions imposed through projection.
2. A second-order staggered-grid discretization with interior pressure
   unknowns and staggered velocity unknowns.

For each grid resolution, the script constructs the corresponding sparse
semi-discrete acoustic operator and computes the numerical eigenvalue nearest
the analytical mode

    p(x, y) = sin(pi*x/Lx) sin(pi*y/Ly),

whose associated positive-imaginary eigenvalue is

    lambda = i*pi*sqrt(1/Lx**2 + 1/Ly**2).

The eigenvalue errors of the SBP and staggered-grid methods are then compared
under mesh refinement.

Main Components
---------------
- sbp2_1d:
    Constructs a second-order diagonal-norm SBP first-derivative operator.

- D_1d_stagg_sparse:
    Constructs a one-dimensional staggered-grid first-derivative operator.

- SBP_proj:
    Builds the projection enforcing homogeneous pressure boundary conditions.

- build_sbp_projected_operator:
    Constructs the two-dimensional projected SBP acoustic operator.

- build_staggered_operator:
    Constructs the two-dimensional staggered-grid acoustic operator.

- compute_eigenvalue_errors:
    Computes the numerical eigenvalue errors for a sequence of grid sizes.

- plot_eigenvalue_convergence:
    Generates a log-log convergence plot and illustrates the approximately
    constant error-factor separation between the two discretizations.

- write_pgfplots_convergence_figure:
    Writes an editable PGFPlots representation of the convergence figure.

Outputs
-------
The script generates:
- eig_vals_conv_factor_shift.pdf
- eig_vals_conv_factor_shift.png
- eig_vals_conv_factor_shift.tex
- eig_vals_conv_final.tex

Dependencies
------------
- NumPy 1.26.4
- SciPy 1.17.0
- Matplotlib 3.7.5
- pathlib
- tikzplotlib 0.10.1 (optional) 

Author
------
Adam Kautzky

"""



import numpy as np
import scipy.sparse as sp
from scipy.sparse import bmat, csc_matrix, diags, eye, kron, vstack
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib as mpl



# ------------------------------------------------------------
# Plotting settings
# ------------------------------------------------------------

mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 400,
    "figure.figsize": (6.2, 4.2),
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,

    "font.size": 10,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,

    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,

    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,

    "lines.linewidth": 1.8,
    "lines.markersize": 5.5,

    "axes.grid": True,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.25,
})


# ------------------------------------------------------------
# Second-order centred SBP operator
# ------------------------------------------------------------

def sbp2_1d(m: int, h: float):
    """
    Constructs second-order diagonal-norm SBP first-derivative operator
    """
    H_diag = np.ones(m)
    H_diag[0] = 0.5
    H_diag[-1] = 0.5

    H = diags(H_diag, 0, format="csc") * h
    HI = diags(1.0 / (H_diag * h), 0, format="csc")

    e = np.ones(m - 1)
    Q = diags(
        [-0.5 * e, 0.5 * e],
        [-1, 1],
        shape=(m, m),
        format="lil",
    ) 

    # Boundary rows
    Q[0, 0] = -0.5
    Q[0, 1] = 0.5
    Q[-1, -2] = -0.5
    Q[-1, -1] = 0.5

    Q = Q.tocsc()
    D = HI @ Q

    return H, HI, D


# ------------------------------------------------------------
# Staggered grid derivative operator
# ------------------------------------------------------------

def D_1d_stagg_sparse(N: int, h: float):
    """
    Constructs first derivative staggered operator
    """
    n_v = N - 1
    n_p = N - 2

    diagonals = [-np.ones(n_p), np.ones(n_p)]
    offsets = [-1, 0]

    D = diags(
        diagonals,
        offsets,
        shape=(n_v, n_p),
        format="csc",
    )

    return D / h


# ------------------------------------------------------------
# SBP projection to enforce homogeneuous BC
# ------------------------------------------------------------

def SBP_proj(m: int, HI_bar: sp.csc_matrix, Ah: sp.csc_matrix):
    """
    Construct the projection enforcing p = 0 on the boundary and return
    the projected SBP operator
    """
    Ntot = m * m
    N = 3 * Ntot

    Im = eye(m, format="csc")
    Im_bar = Im[1:-1, :]  # exclude corners on each edge

    e_l = csc_matrix(([1.0], ([0], [0])), shape=(1, m))
    e_r = csc_matrix(([1.0], ([0], [m - 1])), shape=(1, m))

    # Boundary face extractors
    eW = kron(e_l, Im_bar, format="csc")
    eE = kron(e_r, Im_bar, format="csc")
    eS = kron(Im_bar, e_l, format="csc")
    eN = kron(Im_bar, e_r, format="csc")

    # Corner extractors
    eWS = kron(e_l, e_l, format="csc")
    eES = kron(e_r, e_l, format="csc")
    eWN = kron(e_l, e_r, format="csc")
    eEN = kron(e_r, e_r, format="csc")

    # Extract the pressure block from [vx; vy; p]
    e_p = csc_matrix([[0.0, 0.0, 1.0]])

    L = vstack(
        (
            kron(e_p, eW, format="csc"),
            kron(e_p, eE, format="csc"),
            kron(e_p, eS, format="csc"),
            kron(e_p, eN, format="csc"),
            kron(e_p, eWS, format="csc"),
            kron(e_p, eES, format="csc"),
            kron(e_p, eWN, format="csc"),
            kron(e_p, eEN, format="csc"),
        ),
        format="csc",
    )

    I_N = eye(N, format="csc")

    small_boundary_matrix = (L @ HI_bar @ L.T).tocsc()

    P = (
        I_N
        - (HI_bar @ L.T)
        @ spla.inv(small_boundary_matrix)
        @ L
    ).tocsc()

    Ap = (P @ Ah @ P).tocsc()

    return P, Ap


# ------------------------------------------------------------
# SBP acoustic operator
# ------------------------------------------------------------

def build_sbp_projected_operator(m: int, h: float):
    """
    Build the 2D first-order acoustic SBP-projection operator
    """
    Hx, _, D1x = sbp2_1d(m, h)
    Hy, _, D1y = sbp2_1d(m, h)

    Im = eye(m, format="csc")
    I_3 = eye(3, format="csc")

    Ntot = m * m

    Dx = kron(D1x, Im, format="csc")
    Dy = kron(Im, D1y, format="csc")

    H2 = kron(Hx, Hy, format="csc")
    H2I = diags(1.0 / H2.diagonal(), 0, format="csc")

    HI_bar = kron(I_3, H2I, format="csc")

    Gx = Dx
    Gy = Dy

    Divx = -(H2I @ (Gx.T @ H2))
    Divy = -(H2I @ (Gy.T @ H2))

    Z = csc_matrix((Ntot, Ntot))

    Ah = bmat(
        [
            [Z,     Z,    -Gx],
            [Z,     Z,    -Gy],
            [-Divx, -Divy, Z],
        ],
        format="csc",
    )

    _, Ap = SBP_proj(m, HI_bar, Ah)

    return Ap


# ------------------------------------------------------------
# Staggered-grid acoustic operator
# ------------------------------------------------------------

def build_staggered_operator(m: int, h: float):
    """
    Build the 2D acoustic staggered-grid operator
    """
    Nx = m
    Ny = m

    Nx_int = Nx - 2
    Ny_int = Ny - 2

    Np = Nx_int * Ny_int
    Nvx = (Nx - 1) * Ny_int
    Nvy = Nx_int * (Ny - 1)

    Dx_stagg = D_1d_stagg_sparse(Nx, h)
    Dy_stagg = D_1d_stagg_sparse(Ny, h)

    I_x_stagg = eye(Nx_int, format="csc")
    I_y_stagg = eye(Ny_int, format="csc")

    Gx_stagg = kron(Dx_stagg, I_y_stagg, format="csc")
    Gy_stagg = kron(I_x_stagg, Dy_stagg, format="csc")

    Hp_stagg = (h * h) * eye(Np, format="csc")
    Hvx_stagg = (h * h) * eye(Nvx, format="csc")
    Hvy_stagg = (h * h) * eye(Nvy, format="csc")

    Hp_inv_stagg = diags(
        1.0 / Hp_stagg.diagonal(),
        0,
        format="csc",
    )

    Divx_stagg = -(Hp_inv_stagg @ (Gx_stagg.T @ Hvx_stagg)) #uses weighted adjoint relation between div and grad
    Divy_stagg = -(Hp_inv_stagg @ (Gy_stagg.T @ Hvy_stagg))

    Ah_stagg = bmat(
        [
            [None,       None,       -Gx_stagg],
            [None,       None,       -Gy_stagg],
            [-Divx_stagg, -Divy_stagg, None],
        ],
        format="csc",
    )

    return Ah_stagg


# ------------------------------------------------------------
# Eigenvalue convergence expeirment
# ------------------------------------------------------------

def compute_eigenvalue_errors(m_list, Lx: float = 1.0, Ly: float = 1.0, k_eigs: int = 30):
    """
    Compute eigenvalue errors for the SBP-projection and staggered-grid
    acoustic discretizations
    """
    omega = np.pi * np.sqrt(1.0 / Lx**2 + 1.0 / Ly**2)
    lambda_a = 1j * omega

    h_arr = []
    err_stagg = []
    err_sbp = []

    for m in m_list:
        h = Lx / (m - 1)
        h_arr.append(h)

        print(f"Computing m={m}, h={h:.8e}")

        A_stagg = build_staggered_operator(m, h)
        A_sbp = build_sbp_projected_operator(m, h)

        vals_stagg, _ = spla.eigs(
            A_stagg,
            k=k_eigs,
            sigma=lambda_a,
            which="LM",
            tol=1.0e-12,
        )

        vals_sbp, _ = spla.eigs(
            A_sbp,
            k=k_eigs,
            sigma=lambda_a,
            which="LM",
            tol=1.0e-12,
        )

        lambda_h_stagg = vals_stagg[np.argmin(np.abs(vals_stagg - lambda_a))]
        lambda_h_sbp = vals_sbp[np.argmin(np.abs(vals_sbp - lambda_a))]

        e_stagg = float(np.abs(lambda_h_stagg - lambda_a))
        e_sbp = float(np.abs(lambda_h_sbp - lambda_a))

        err_stagg.append(e_stagg)
        err_sbp.append(e_sbp)

        print(f"  lambda analytical = {lambda_a}")
        print(f"  lambda staggered   = {lambda_h_stagg}")
        print(f"  lambda SBP         = {lambda_h_sbp}")
        print(f"  error staggered    = {e_stagg:.8e}")
        print(f"  error SBP          = {e_sbp:.8e}")
        print(f"  staggered / SBP    = {e_stagg / e_sbp:.6f}")
        print()

    return (
        np.asarray(h_arr, dtype=float),
        np.asarray(err_stagg, dtype=float),
        np.asarray(err_sbp, dtype=float),
        lambda_a,
    )


# ------------------------------------------------------------
# PLOTTING - generated with ChatGPT-5.5
# ------------------------------------------------------------

def plot_eigenvalue_convergence(
    h: np.ndarray,
    err_stagg: np.ndarray,
    err_sbp: np.ndarray,
    savepath_pdf: str = "eig_vals_conv_factor_shift.pdf",
    savepath_png: str = "eig_vals_conv_factor_shift.png",
    savepath_tikz: str | None = "eig_vals_conv_factor_shift.tex"
    ):
    """
    Plot eigenvalue convergence and illustrate:
      - second-order slopes beside each line,
      - approximately constant vertical factor separation.
    """
    order = np.argsort(h)
    h = h[order]
    err_stagg = err_stagg[order]
    err_sbp = err_sbp[order]

    slope_stagg = np.polyfit(np.log(h), np.log(err_stagg), 1)[0]
    slope_sbp = np.polyfit(np.log(h), np.log(err_sbp), 1)[0]

    ratio = err_stagg / err_sbp
    ratio_geom = float(np.exp(np.mean(np.log(ratio))))

    print("Summary")
    print("-------")
    print("h values             =", h)
    print("staggered errors     =", err_stagg)
    print("SBP errors           =", err_sbp)
    print("staggered / SBP      =", ratio)
    print(f"staggered slope      = {slope_stagg:.8f}")
    print(f"SBP slope            = {slope_sbp:.8f}")
    print(f"geometric mean ratio = {ratio_geom:.8f}")

    # Use exact 4x guide only when numerically justified.
    if np.allclose(ratio, 4.0, rtol=0.15):
        guide_factor = 4.0
        factor_text = r"$\approx 4\times$"
        guide_label = r"$4e_{\lambda,\mathrm{SBP}}$"
    else:
        guide_factor = ratio_geom
        factor_text = rf"$\approx {ratio_geom:.2f}\times$"
        guide_label = rf"${ratio_geom:.2f}e_{{\lambda,\mathrm{{SBP}}}}$"

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    ax.loglog(
        h,
        err_stagg,
        "-o",
        label="Staggered grid",
    )

    ax.loglog(
        h,
        err_sbp,
        "-s",
        label="SBP projection",
    )

    ax.loglog(
        h,
        guide_factor * err_sbp,
        "--",
        linewidth=1.3,
        label=guide_label,
    )

    # --------------------------------------------------------
    # Slope labels beside the lines
    # --------------------------------------------------------
    # Choose the middle interval for placing text.
    # This avoids placing text too close to endpoints.
    if len(h) >= 3:
        i_label = len(h) // 2
    else:
        i_label = 0

    x_label = h[i_label]

    # Multiplicative offsets are best on log axes.
    y_sbp_label = err_sbp[i_label] * 1.35      # above SBP
    y_stagg_label = err_stagg[i_label] / 1.35  # below STAGG

    ax.text(
        x_label,
        y_sbp_label,
        rf"slope ${slope_sbp:.2f}$",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    ax.text(
        x_label,
        y_stagg_label,
        rf"slope ${slope_stagg:.2f}$",
        ha="center",
        va="top",
        fontsize=10,
    )

    # --------------------------------------------------------
    # Factor-of-four vertical separation annotation
    # --------------------------------------------------------
    j = len(h) // 2
    x_arrow = h[j]
    y_low = err_sbp[j]
    y_high = err_stagg[j]

    if y_high < y_low:
        y_low, y_high = y_high, y_low

    ax.annotate(
        "",
        xy=(x_arrow, y_high),
        xytext=(x_arrow, y_low),
        arrowprops={
            "arrowstyle": "<->",
            "linewidth": 1.2,
        },
    )

    ax.annotate(
        factor_text,
        xy=(x_arrow, np.sqrt(y_low * y_high)),
        xytext=(11, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=10,
    )

    ax.set_xlabel(r"Grid spacing $h$")
    ax.set_ylabel(r"Eigenvalue error $e_\lambda = |\lambda_h-\lambda|$")
    ax.grid(True, which="both", alpha=0.25)

    # Keep legend simple because slope labels are now on the curves.
    ax.legend(frameon=False, loc="best")

    fig.tight_layout()

    fig.savefig(savepath_pdf)
    fig.savefig(savepath_png)

    if savepath_tikz is not None:
        try:
            import tikzplotlib
            tikzplotlib.save(
                savepath_tikz,
                figure=fig,
                axis_width=r"0.72\textwidth",
                axis_height=r"0.48\textwidth",
            )
            print(f"Saved TikZ/PGFPlots file to {savepath_tikz}")
        except Exception as exc:
            print(f"Could not save tikzplotlib output: {exc}")

    plt.show()
    plt.close(fig)


from pathlib import Path


def write_pgfplots_convergence_figure(
    h: np.ndarray,
    err_stagg: np.ndarray,
    err_sbp: np.ndarray,
    output_path: str = "eig_vals_conv_final.tex",
):
    """
    Write an editable PGFPlots figure directly from the computed convergence
    data. This avoids tikzplotlib/Matplotlib compatibility problems.
    """
    order = np.argsort(h)
    h = np.asarray(h, dtype=float)[order]
    err_stagg = np.asarray(err_stagg, dtype=float)[order]
    err_sbp = np.asarray(err_sbp, dtype=float)[order]

    slope_stagg = np.polyfit(np.log(h), np.log(err_stagg), 1)[0]
    slope_sbp = np.polyfit(np.log(h), np.log(err_sbp), 1)[0]

    ratio = err_sbp / err_stagg
    ratio_geom = float(np.exp(np.mean(np.log(ratio))))

    # Show an exact 4x guide only when supported by the computed values.
    if np.allclose(ratio, 4.0, rtol=0.15):
        guide_factor = 4.0
        factor_text = r"\approx 4\!\times"
        guide_legend = r"$4e_{\lambda,\mathrm{STAGG}}$"
    else:
        guide_factor = ratio_geom
        factor_text = rf"\approx {ratio_geom:.2f}\!\times"
        guide_legend = rf"${ratio_geom:.2f}e_{{\lambda,\mathrm{{STAGG}}}}$"

    # Use a middle data point for annotations.
    j = len(h) // 2
    x_arrow = h[j]
    y_lower = err_stagg[j]
    y_upper = err_sbp[j]

    # Place slope labels along the curves, away from the arrow.
    x_label = h[max(j - 1, 0)]
    y_label_stagg = err_stagg[max(j - 1, 0)] / 1.30
    y_label_sbp = err_sbp[max(j - 1, 0)] * 1.30

    stagg_coordinates = "\n".join(
        f"        ({x:.16e}, {y:.16e})"
        for x, y in zip(h, err_stagg)
    )

    sbp_coordinates = "\n".join(
        f"        ({x:.16e}, {y:.16e})"
        for x, y in zip(h, err_sbp)
    )

    guide_coordinates = "\n".join(
        f"        ({x:.16e}, {guide_factor * y:.16e})"
        for x, y in zip(h, err_stagg)
    )

    tex = rf"""% Automatically generated PGFPlots figure.
% The data are produced by the eigenvalue convergence calculation.
% Edit node positions, line styles, or legend placement directly here.

\begin{{tikzpicture}}
\begin{{loglogaxis}}[
    width=0.76\textwidth,
    height=0.53\textwidth,
    xlabel={{Grid spacing $h$}},
    ylabel={{Eigenvalue error $e_\lambda = |\lambda_h-\lambda|$}},
    grid=both,
    minor grid style={{gray!15}},
    major grid style={{gray!30}},
    legend style={{
        draw=none,
        fill=none,
        font=\small,
        at={{(0.03,0.97)}},
        anchor=north west
    }},
    tick label style={{font=\small}},
    label style={{font=\small}},
    clip=false,
]

\addplot+[
    solid,
    mark=*,
    thick,
]
coordinates {{
{stagg_coordinates}
}}
node[pos=0.18, below, yshift=-3pt, font=\small]
{{$\mathcal{{O}}(h^{{{slope_stagg:.3f}}})$}};
\addlegendentry{{Staggered grid}}

\addplot+[
    solid,
    mark=square*,
    thick,
]
coordinates {{
{sbp_coordinates}
}}
node[pos=0.18, above, yshift=3pt, font=\small]
{{$\mathcal{{O}}(h^{{{slope_sbp:.3f}}})$}};
\addlegendentry{{SBP projection}}

\addplot+[
    dashed,
    no marks,
    thick,
]
coordinates {{
{guide_coordinates}
}};
\addlegendentry{{{guide_legend}}}

\draw[<->, thick]
    (axis cs:{x_arrow:.16e},{y_lower:.16e})
    --
    node[midway, right, xshift=3pt, font=\small]
    {{$ {factor_text} $}}
    (axis cs:{x_arrow:.16e},{y_upper:.16e});

\end{{loglogaxis}}
\end{{tikzpicture}}
"""

    Path(output_path).write_text(tex, encoding="utf-8")

    print(f"Saved editable PGFPlots figure to: {output_path}")
    print(f"SBP / staggered ratios: {ratio}")
    print(f"Geometric mean ratio: {ratio_geom:.8f}")
    print(f"Staggered slope: {slope_stagg:.8f}")
    print(f"SBP slope: {slope_sbp:.8f}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    m_list = [10, 20, 40, 80, 160]

    h, err_stagg, err_sbp, lambda_a = compute_eigenvalue_errors(
        m_list=m_list,
        Lx=1.0,
        Ly=1.0,
        k_eigs=30,
    )

    print(f"Target eigenvalue: {lambda_a}")
    print()
    write_pgfplots_convergence_figure(
    h=h,
    err_stagg=err_stagg,
    err_sbp=err_sbp,
    output_path="eig_vals_conv_final.tex",
    )
    plot_eigenvalue_convergence(
        h=h,
        err_stagg=err_stagg,
        err_sbp=err_sbp,
        savepath_pdf="eig_vals_conv_factor_shift.pdf",
        savepath_png="eig_vals_conv_factor_shift.png",
        savepath_tikz = "eig_vals_conv_factor_shift.tex"
    )


if __name__ == "__main__":
    main()