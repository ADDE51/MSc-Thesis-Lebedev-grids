"""
3D Elastic Wave Solver on a Staggered grid
=========================================================================

Purpose
-------
This script uses the binary grid/cluster notation developed in the report, to solve the 3D elastic wave 
equation in a homogeneous medium using the Virieux staggered grid scheme. This implementation uses 3D
Numpy arrays and Numba-accelerated finite-difference kernels for the spatial derivative and explicit updates.



Outputs
-------
Depending on the enabled options, the script may generate:

- PNG snapshots of the velocity field
- central xy/xz/yz plane plots
- 3D intersecting plane plots
- Christoffel cut plots
- source-spectrum diagnostics


Dependencies
------------
- NumPy 1.26.4
- Numba 0.65.1
- SciPy 1.17.0 (optional)
- Matplotlib 3.7.5

Notes
-----
The simulation routine terminates after saving the selected snapshot, so returned fields correspond
to that snapshot time unless the 'break' statement is removed.

Author
------
Adam Kautzky
"""


import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use("Agg")



# ------------------------------------------------------------
# Global component conventions
# ------------------------------------------------------------

Bit3 = tuple[int, int, int]

X, Y, Z = 0, 1, 2
VX, VY, VZ = 0, 1, 2
XX, YY, ZZ, YZ, XZ, XY = 0, 1, 2, 3, 4, 5

DX = DY = DZ = 0.0

VCOMP_NAMES = ['x', 'y', 'z']
SCOMP_NAMES = ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']

SCLUSTERS = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
]

VCLUSTERS = [
    (1, 1, 1),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
]

V_SHIFT = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
]

S_SHIFT = [
    (1, 1, 1),
    (1, 1, 1),
    (1, 1, 1),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
]

MATERIAL_GRIDS = [
    (1, 1, 1),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
]

# ------------------------------------------------------------
# Binary cluster and staggered-grid representation
# ------------------------------------------------------------

def xor3(a: Bit3, b: Bit3) -> Bit3:
    return (a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2])


def toggle_axis(g: Bit3, axis: int) -> Bit3:
    if axis == X:
        return (g[0] ^ 1, g[1], g[2])
    if axis == Y:
        return (g[0], g[1] ^ 1, g[2])
    if axis == Z:
        return (g[0], g[1], g[2] ^ 1)
    raise ValueError("axis must be 0, 1, or 2")


def shape_from_grid(g: Bit3, Nx: int, Ny: int, Nz: int) -> tuple[int, int, int]:
    return (
        Nx if g[0] == 0 else Nx - 1,
        Ny if g[1] == 0 else Ny - 1,
        Nz if g[2] == 0 else Nz - 1,
    )


def velocity_grid(vcluster: Bit3, comp: int) -> Bit3:
    return xor3(vcluster, V_SHIFT[comp])


def stress_grid(scluster: Bit3, comp: int) -> Bit3:
    return xor3(scluster, S_SHIFT[comp])




# ------------------------------------------------------------
# Numba-acclerated derivative kernels
# ------------------------------------------------------------

@njit(parallel=True, fastmath=False)
def diff_x_p2d(u, h): #diff in x, primary to dual
    nx, ny, nz = u.shape
    out = np.empty((nx - 1, ny, nz), dtype=u.dtype)
    ih = 1.0 / h
    for i in prange(nx - 1):
        for j in range(ny):
            for k in range(nz):
                out[i, j, k] = (u[i + 1, j, k] - u[i, j, k]) * ih
    return out


@njit(parallel=True, fastmath=False)
def diff_y_p2d(u, h): #diff in y, primary to dual
    nx, ny, nz = u.shape
    out = np.empty((nx, ny - 1, nz), dtype=u.dtype)
    ih = 1.0 / h
    for i in prange(nx):
        for j in range(ny - 1):
            for k in range(nz):
                out[i, j, k] = (u[i, j + 1, k] - u[i, j, k]) * ih
    return out


@njit(parallel=True, fastmath=False)
def diff_z_p2d(u, h): #diff in z, primary to dual
    nx, ny, nz = u.shape
    out = np.empty((nx, ny, nz - 1), dtype=u.dtype)
    ih = 1.0 / h
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz - 1):
                out[i, j, k] = (u[i, j, k + 1] - u[i, j, k]) * ih
    return out


@njit(parallel=True, fastmath=False)
def diff_x_d2p(u, h): #diff in x, dual to primary
    nx, ny, nz = u.shape
    out = np.empty((nx + 1, ny, nz), dtype=u.dtype)
    ih = 1.0 / h
    for j in prange(ny):
        for k in range(nz):
            out[0, j, k] = u[0, j, k] * ih
            out[nx, j, k] = -u[nx - 1, j, k] * ih
    for i in prange(1, nx):
        for j in range(ny):
            for k in range(nz):
                out[i, j, k] = (u[i, j, k] - u[i - 1, j, k]) * ih
    return out


@njit(parallel=True, fastmath=False)
def diff_y_d2p(u, h): #diff in y, dual to primary
    nx, ny, nz = u.shape
    out = np.empty((nx, ny + 1, nz), dtype=u.dtype)
    ih = 1.0 / h
    for i in prange(nx):
        for k in range(nz):
            out[i, 0, k] = u[i, 0, k] * ih
            out[i, ny, k] = -u[i, ny - 1, k] * ih
    for i in prange(nx):
        for j in range(1, ny):
            for k in range(nz):
                out[i, j, k] = (u[i, j, k] - u[i, j - 1, k]) * ih
    return out


@njit(parallel=True, fastmath=False)
def diff_z_d2p(u, h): #diff in z, dual to primary
    nx, ny, nz = u.shape
    out = np.empty((nx, ny, nz + 1), dtype=u.dtype)
    ih = 1.0 / h
    for i in prange(nx):
        for j in range(ny):
            out[i, j, 0] = u[i, j, 0] * ih
            out[i, j, nz] = -u[i, j, nz - 1] * ih
    for i in prange(nx):
        for j in range(ny):
            for k in range(1, nz):
                out[i, j, k] = (u[i, j, k] - u[i, j, k - 1]) * ih
    return out


def diff_axis(u: np.ndarray, grid: Bit3, axis: int, h: float): #choose diff func based on grid
    bit = grid[axis]
    if axis == X:
        return (diff_x_p2d(u, h), toggle_axis(grid, axis)) if bit == 0 else (diff_x_d2p(u, h), toggle_axis(grid, axis))
    if axis == Y:
        return (diff_y_p2d(u, h), toggle_axis(grid, axis)) if bit == 0 else (diff_y_d2p(u, h), toggle_axis(grid, axis))
    if axis == Z:
        return (diff_z_p2d(u, h), toggle_axis(grid, axis)) if bit == 0 else (diff_z_d2p(u, h), toggle_axis(grid, axis))
    raise ValueError("axis must be X, Y, or Z")



@njit(parallel=True, fastmath=False) #addition of two comps
def add2(a, b):
    nx, ny, nz = a.shape
    out = np.empty_like(a)
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                out[i, j, k] = a[i, j, k] + b[i, j, k]
    return out


@njit(parallel=True, fastmath=False)
def add3(a, b, c): #addition of three comps
    nx, ny, nz = a.shape
    out = np.empty_like(a)
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                out[i, j, k] = a[i, j, k] + b[i, j, k] + c[i, j, k]
    return out


@njit(parallel=True, fastmath=False)
def axpy_inplace(y, a, x): #accelerated y <- y + a*x operation
    nx, ny, nz = y.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                y[i, j, k] += a * x[i, j, k]


# ------------------------------------------------------------
# Build staggered state
# ------------------------------------------------------------

def zeros_on_grid(grid: Bit3, Nx: int, Ny: int, Nz: int) -> np.ndarray: #allocate a zero-valued field on a selected grid
    return np.zeros(shape_from_grid(grid, Nx, Ny, Nz), dtype=np.float64)


def build_state(SCL: Bit3, Nx: int, Ny: int, Nz: int): #allocate stress and velocity state arrays for one cluster pair
    VCL = xor3(SCL, (1,1,1)) #bitwise complement of stress cluster
 
    V = [zeros_on_grid(velocity_grid(VCL, comp), Nx, Ny, Nz) for comp in range(3)]
    S = [zeros_on_grid(stress_grid(SCL, comp), Nx, Ny, Nz) for comp in range(6)]
   
    return V, S


def coords_for_grid(grid: Bit3, Nx: int, Ny: int, Nz: int, Lx: float, Ly: float, Lz: float): 
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dz = Lz / (Nz - 1)
    if grid[0] == 0:
        x = np.linspace(0.0, Lx, Nx)
    else:
        x = (np.arange(Nx - 1) + 0.5) * dx
    if grid[1] == 0:
        y = np.linspace(0.0, Ly, Ny)
    else:
        y = (np.arange(Ny - 1) + 0.5) * dy
    if grid[2] == 0:
        z = np.linspace(0.0, Lz, Nz)
    else:
        z = (np.arange(Nz - 1) + 0.5) * dz
    return x, y, z


# ------------------------------------------------------------
# Sources
# ------------------------------------------------------------

def gaussian_on_grid(grid: Bit3, Nx: int, Ny: int, Nz: int, Lx: float, Ly: float, Lz: float, x0: float, y0: float, z0: float, width: float):
    x, y, z = coords_for_grid(grid, Nx, Ny, Nz, Lx, Ly, Lz)
    Xg, Yg, Zg = np.meshgrid(x, y, z, indexing="ij")
    return np.exp(-((Xg - x0)**2 + (Yg - y0)**2 + (Zg - z0)**2) / (2.0 * width**2))


def build_shear_xz_source(
    SCL: Bit3,
    Nx: int, Ny: int, Nz: int,
    Lx: float, Ly: float, Lz: float,
    width: float,
):
    x0, y0, z0 = Lx / 2.0, Ly / 2.0, Lz / 2.0

    Q = []
    for comp in range(6):
        g = stress_grid(SCL, comp)
        if comp == XZ:
            Q.append(
                gaussian_on_grid(
                    g, Nx, Ny, Nz, Lx, Ly, Lz,
                    x0, y0, z0, width
                )
            )
        else:
            Q.append(
                np.zeros(shape_from_grid(g, Nx, Ny, Nz), dtype=np.float64)
            )

    return Q


def build_explosive_source(
    SCL: Bit3,
    Nx: int, Ny: int, Nz: int,
    Lx: float, Ly: float, Lz: float,
    width: float,
):
    x0, y0, z0 = Lx / 2.0, Ly / 2.0, Lz / 2.0

    Q = []
    for comp in range(6):
        g = stress_grid(SCL, comp)
        if comp in (XX, YY, ZZ):
            Q.append(
                gaussian_on_grid(
                    g, Nx, Ny, Nz, Lx, Ly, Lz,
                    x0, y0, z0, width
                )
            )
        else:
            Q.append(
                np.zeros(shape_from_grid(g, Nx, Ny, Nz), dtype=np.float64)
            )

    return Q


def ricker(t: float, f0: float, t0: float) -> float:
    a = np.pi * f0 * (t - t0)
    return (1.0 - 2.0 * a*a) * np.exp(-a*a)



# ------------------------------------------------------------
# Compute fields and update
# ------------------------------------------------------------

def compute_strain(V, SCL: Bit3):
    VCL = xor3(SCL, (1, 1, 1))

    vx, vy, vz = V[VX], V[VY], V[VZ]

    gvx = velocity_grid(VCL, VX)
    gvy = velocity_grid(VCL, VY)
    gvz = velocity_grid(VCL, VZ)

    exx, g_exx = diff_axis(vx, gvx, X, DX)
    eyy, g_eyy = diff_axis(vy, gvy, Y, DY)
    ezz, g_ezz = diff_axis(vz, gvz, Z, DZ)

    if not (g_exx == g_eyy == g_ezz):
        raise RuntimeError("Normal strain components are not on the same grid")

    dvy_dz, g1 = diff_axis(vy, gvy, Z, DZ)
    dvz_dy, g2 = diff_axis(vz, gvz, Y, DY)
    if g1 != g2:
        raise RuntimeError("Grid mismatch in eyz")
    eyz = add2(dvy_dz, dvz_dy)

    dvx_dz, g1 = diff_axis(vx, gvx, Z, DZ)
    dvz_dx, g2 = diff_axis(vz, gvz, X, DX)
    if g1 != g2:
        raise RuntimeError("Grid mismatch in exz")
    exz = add2(dvx_dz, dvz_dx)

    dvx_dy, g1 = diff_axis(vx, gvx, Y, DY)
    dvy_dx, g2 = diff_axis(vy, gvy, X, DX)
    if g1 != g2:
        raise RuntimeError("Grid mismatch in exy")
    exy = add2(dvx_dy, dvy_dx)

    return [exx, eyy, ezz, eyz, exz, exy]


def apply_material_single_cluster(strain, C):
    
    exx, eyy, ezz, eyz, exz, exy = strain

    # Normal block: [sxx, syy, szz]^T = C_normal [exx, eyy, ezz]^T
    sxx = C[0, 0] * exx + C[0, 1] * eyy + C[0, 2] * ezz
    syy = C[1, 0] * exx + C[1, 1] * eyy + C[1, 2] * ezz
    szz = C[2, 0] * exx + C[2, 1] * eyy + C[2, 2] * ezz

    # Diagonal shear terms
    syz = C[3, 3] * eyz
    sxz = C[4, 4] * exz
    sxy = C[5, 5] * exy

    return [sxx, syy, szz, syz, sxz, sxy]



def update_stresses(S, stress_rhs, dt: float):
    for comp in range(6):
        axpy_inplace(S[comp], dt, stress_rhs[comp])


def update_velocities(V, S, SCL: Bit3, rho: float, dt: float):
    VCL = xor3(SCL, (1, 1, 1))
    scale = dt / rho

    sxx, syy, szz = S[XX], S[YY], S[ZZ]
    syz, sxz, sxy = S[YZ], S[XZ], S[XY]

    gsxx = stress_grid(SCL, XX)
    gsyy = stress_grid(SCL, YY)
    gszz = stress_grid(SCL, ZZ)
    gsyz = stress_grid(SCL, YZ)
    gsxz = stress_grid(SCL, XZ)
    gsxy = stress_grid(SCL, XY)

    # vx update: d_x sigma_xx + d_y sigma_xy + d_z sigma_xz
    a, ga = diff_axis(sxx, gsxx, X, DX)
    b, gb = diff_axis(sxy, gsxy, Y, DY)
    c, gc = diff_axis(sxz, gsxz, Z, DZ)

    gvx = velocity_grid(VCL, VX)
    if not (ga == gb == gc == gvx):
        raise RuntimeError(f"Grid mismatch in vx update: {ga}, {gb}, {gc}, expected {gvx}")

    #use numba kernels
    rhs_vx = add3(a, b, c)
    axpy_inplace(V[VX], scale, rhs_vx)

    # vy update: d_x sigma_xy + d_y sigma_yy + d_z sigma_yz
    a, ga = diff_axis(sxy, gsxy, X, DX)
    b, gb = diff_axis(syy, gsyy, Y, DY)
    c, gc = diff_axis(syz, gsyz, Z, DZ)

    gvy = velocity_grid(VCL, VY)
    if not (ga == gb == gc == gvy):
        raise RuntimeError(f"Grid mismatch in vy update: {ga}, {gb}, {gc}, expected {gvy}")

    rhs_vy = add3(a, b, c)
    axpy_inplace(V[VY], scale, rhs_vy)

    # vz update: d_x sigma_xz + d_y sigma_yz + d_z sigma_zz
    a, ga = diff_axis(sxz, gsxz, X, DX)
    b, gb = diff_axis(syz, gsyz, Y, DY)
    c, gc = diff_axis(szz, gszz, Z, DZ)

    gvz = velocity_grid(VCL, VZ)
    if not (ga == gb == gc == gvz):
        raise RuntimeError(f"Grid mismatch in vz update: {ga}, {gb}, {gc}, expected {gvz}")

    rhs_vz = add3(a, b, c)
    axpy_inplace(V[VZ], scale, rhs_vz)


def add_source_to_strain(strain, Q, pulse: float):
    if Q is None:
        return

    for comp in range(6):
        if Q[comp] is not None:
            axpy_inplace(strain[comp], pulse, Q[comp])
            

def step_leapfrog(V, S, Q, C, SCL: Bit3, rho: float, dt: float, pulse: float):
    strain_rhs = compute_strain(V, SCL)
    add_source_to_strain(strain_rhs, Q, pulse)

    stress_rhs = apply_material_single_cluster(strain_rhs, C)

    update_stresses(S, stress_rhs, dt)
    update_velocities(V, S, SCL, rho, dt)




# ------------------------------------------------------------
# Build 6x6 Stiffness matrices for different media
# ------------------------------------------------------------

def make_isotropic_C(rho: float, VP: float, VS: float) -> np.ndarray:
    mu = rho * VS**2
    lam = rho * (VP**2 - 2.0 * VS**2)
    C11 = lam + 2.0 * mu
    C12 = lam
    C44 = mu
    return np.array([
        [C11, C12, C12, 0.0, 0.0, 0.0],
        [C12, C11, C12, 0.0, 0.0, 0.0],
        [C12, C12, C11, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, C44, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, C44, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, C44],
    ], dtype=np.float64)


def make_vti_C(rho: float, VP: float, VS: float) -> np.ndarray:
    epsilon = 0.334
    gamma = 0.575
    delta_star = 0.93
    C33 = rho * VP**2
    C44 = rho * VS**2
    C11 = C33 * (1.0 + 2.0 * epsilon)
    C66 = C44 * (1.0 + 2.0 * gamma)
    C12 = C11 - 2.0 * C66
    C13 = C33 * delta_star - C44
    return np.array([
        [C11, C12, C13, 0.0, 0.0, 0.0],
        [C12, C11, C13, 0.0, 0.0, 0.0],
        [C13, C13, C33, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, C44, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, C44, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, C66],
    ], dtype=np.float64)


# ------------------------------------------------------------
# Calculate max wave speed from Christoffel eqn.
# ------------------------------------------------------------


def S_sigma(n):
    nx, ny, nz = n
    return np.array([
        [nx, 0.0, 0.0, 0.0, nz, ny],
        [0.0, ny, 0.0, nz, 0.0, nx],
        [0.0, 0.0, nz, ny, nx, 0.0],
    ])

def wave_speeds_direction(C, rho, n):

    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)

    S = S_sigma(n)
    Gamma = (S @ C @ S.T) / rho #Christoffel mat

    # Symmetric eigenvalue solver
    eigvals = np.linalg.eigvalsh(Gamma)

    # Protect against tiny negative roundoff
    eigvals = np.maximum(eigvals, 0.0)

    return np.sqrt(eigvals)


def max_wave_speed(C, rho, ntheta=181, nphi=361):
    
    vmax = 0.0
    best_direction = None
    best_speeds = None

    for i in range(ntheta):
        theta = np.pi * i / (ntheta - 1)
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)

        for j in range(nphi):
            phi = 2.0 * np.pi * j / (nphi - 1)

            n = np.array([
                sin_t * np.cos(phi),
                sin_t * np.sin(phi),
                cos_t,
            ])

            speeds = wave_speeds_direction(C, rho, n)
            cmax_dir = speeds[-1]

            if cmax_dir > vmax:
                vmax = cmax_dir
                best_direction = n
                best_speeds = speeds

    return vmax, best_direction, best_speeds


# ------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------


def plot_three_planes_2d(field3: np.ndarray, Lx: float, Ly: float, Lz: float, title: str, savepath: str | None = None) -> None:
    nx, ny, nz = field3.shape
    ix, iy, iz = nx // 2, ny // 2, nz // 2

    xy = field3[:, :, iz]
    xz = field3[:, iy, :]
    yz = field3[ix, :, :]

  # vmax = max(np.max(np.abs(xy)), np.max(np.abs(xz)), np.max(np.abs(yz)))
    vals = np.concatenate([xy.ravel(), xz.ravel(), yz.ravel()])
    vmax = 0.8*np.percentile(np.abs(vals), 99)
    #vmax = np.percentile(np.abs(field), 99)
    if vmax == 0.0:
        vmax = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    im = axes[0].imshow(xy.T, origin="lower", extent=[0, Lx, 0, Ly], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title(r"$xy$",fontsize=18)
    axes[0].set_xlabel(r"$x$",fontsize=18)
    axes[0].set_ylabel(r"$y$",fontsize=18)

    axes[1].imshow(xz.T, origin="lower", extent=[0, Lx, 0, Lz], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title(r"$xz$",fontsize=18)
    axes[1].set_xlabel(r"$x$",fontsize=18)
    axes[1].set_ylabel(r"$z$",fontsize=18)

    axes[2].imshow(yz.T, origin="lower", extent=[0, Ly, 0, Lz], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[2].set_title(r"$yz$",fontsize=18)
    axes[2].set_xlabel(r"$y$",fontsize=18)
    axes[2].set_ylabel(r"$z$",fontsize=18)

    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle(title)

    if savepath is not None:
        plt.savefig(savepath, dpi=250, bbox_inches="tight")
    #plt.show()
    plt.close(fig)

def plot_lisitsa_pretty_fullplanes(  #generated using ChatGPT-5.5
    field3,
    Lx, Ly, Lz,
    x0=None, y0=None, z0=None,
    cmap="RdBu_r",
    title=None,
    savepath=None,
    elev=22,
    azim=-55,
    show_source=True,
    upsample=1,
    show_plane_borders=True,
    plane_border_color="0.25",
    plane_border_lw=1.4,
    ):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    nx, ny, nz = field3.shape

    if x0 is None:
        x0 = Lx / 2
    if y0 is None:
        y0 = Ly / 2
    if z0 is None:
        z0 = Lz / 2

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    z = np.linspace(0, Lz, nz)

    ix = np.argmin(np.abs(x - x0))
    iy = np.argmin(np.abs(y - y0))
    iz = np.argmin(np.abs(z - z0))

    xy = field3[:, :, iz]
    xz = field3[:, iy, :]
    yz = field3[ix, :, :]

    vals = np.concatenate([xy.ravel(), xz.ravel(), yz.ravel()])
    vmax = np.percentile(np.abs(vals), 99.0)
    if vmax == 0:
        vmax = 1.0

    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap_obj = plt.colormaps[cmap]
    #cmap_obj = cm.get_cmap(cmap)

    def upsample_2d(arr, a, b, factor):
        """
        Upsample only for visualization.
        Uses scipy if available; otherwise leaves unchanged.
        """
        if factor <= 1:
            return arr, a, b

        try:
            from scipy.ndimage import zoom
            arr_u = zoom(arr, factor, order=1)
            a_u = np.linspace(a[0], a[-1], arr_u.shape[0])
            b_u = np.linspace(b[0], b[-1], arr_u.shape[1])
            return arr_u, a_u, b_u
        except Exception:
            return arr, a, b

   
    def make_facecolors(slab):
        FC = cmap_obj(norm(slab))

        alpha = (np.abs(slab) / vmax) ** 0.8
        alpha = np.clip(alpha, 0.0, 1.0)
        alpha[alpha < 0.08] = 0.0

        FC[..., -1] = 0.85 * alpha
        return FC
   
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")

    # ---------- XY plane ----------
    xy_u, x_xy, y_xy = upsample_2d(xy, x, y, upsample)
    X_xy, Y_xy = np.meshgrid(x_xy, y_xy, indexing="ij")
    Z_xy = np.full_like(X_xy, z[iz])
    FC_xy = make_facecolors(xy_u)

    ax.plot_surface(
        X_xy, Y_xy, Z_xy,
        facecolors=FC_xy,
        shade=False,
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
    )

    # ---------- XZ plane ----------
    xz_u, x_xz, z_xz = upsample_2d(xz, x, z, upsample)
    X_xz, Z_xz = np.meshgrid(x_xz, z_xz, indexing="ij")
    Y_xz = np.full_like(X_xz, y[iy])
    FC_xz = make_facecolors(xz_u)

    ax.plot_surface(
        X_xz, Y_xz, Z_xz,
        facecolors=FC_xz,
        shade=False,
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
    )

    # ---------- YZ plane ----------
    yz_u, y_yz, z_yz = upsample_2d(yz, y, z, upsample)
    Y_yz, Z_yz = np.meshgrid(y_yz, z_yz, indexing="ij")
    X_yz = np.full_like(Y_yz, x[ix])
    FC_yz = make_facecolors(yz_u)

    ax.plot_surface(
        X_yz, Y_yz, Z_yz,
        facecolors=FC_yz,
        shade=False,
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
    )

    # ---------- borders around the three plotted planes ----------
    def draw_rect_xy(zc):
        ax.plot([0, Lx], [0, 0],  [zc, zc], color=plane_border_color, lw=plane_border_lw)
        ax.plot([0, Lx], [Ly, Ly], [zc, zc], color=plane_border_color, lw=plane_border_lw)
        ax.plot([0, 0],  [0, Ly], [zc, zc], color=plane_border_color, lw=plane_border_lw)
        ax.plot([Lx, Lx], [0, Ly], [zc, zc], color=plane_border_color, lw=plane_border_lw)

    def draw_rect_xz(yc):
        ax.plot([0, Lx], [yc, yc], [0, 0],  color=plane_border_color, lw=plane_border_lw)
        ax.plot([0, Lx], [yc, yc], [Lz, Lz], color=plane_border_color, lw=plane_border_lw)
        ax.plot([0, 0],  [yc, yc], [0, Lz], color=plane_border_color, lw=plane_border_lw)
        ax.plot([Lx, Lx], [yc, yc], [0, Lz], color=plane_border_color, lw=plane_border_lw)

    def draw_rect_yz(xc):
        ax.plot([xc, xc], [0, Ly], [0, 0],  color=plane_border_color, lw=plane_border_lw)
        ax.plot([xc, xc], [0, Ly], [Lz, Lz], color=plane_border_color, lw=plane_border_lw)
        ax.plot([xc, xc], [0, 0],  [0, Lz], color=plane_border_color, lw=plane_border_lw)
        ax.plot([xc, xc], [Ly, Ly], [0, Lz], color=plane_border_color, lw=plane_border_lw)

    if show_plane_borders:
        draw_rect_xy(z[iz])
        draw_rect_xz(y[iy])
        #draw_rect_yz(x[ix])

    # source marker
    if show_source:
        ax.scatter(
            [x[ix]], [y[iy]], [z[iz]],
            color="k",
            s=28,
            depthshade=False,
            zorder=10,
        )

    # ---------- outer box ----------
    def edge(x1, x2, y1, y2, z1, z2):
        ax.plot([x1, x2], [y1, y2], [z1, z2], color="k", lw=0.8)

    edge(0, Lx, 0, 0, 0, 0)
    edge(0, Lx, Ly, Ly, 0, 0)
    edge(0, 0, 0, Ly, 0, 0)
    edge(Lx, Lx, 0, Ly, 0, 0)

    edge(0, Lx, 0, 0, Lz, Lz)
    edge(0, Lx, Ly, Ly, Lz, Lz)
    edge(0, 0, 0, Ly, Lz, Lz)
    edge(Lx, Lx, 0, Ly, Lz, Lz)

    edge(0, 0, 0, 0, 0, Lz)
    edge(Lx, Lx, 0, 0, 0, Lz)
    edge(0, 0, Ly, Ly, 0, Lz)
    edge(Lx, Lx, Ly, Ly, 0, Lz)

    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(0, Lz)
    ax.set_box_aspect((Lx, Ly, Lz))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)

    ax.view_init(elev=elev, azim=azim)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.72, pad=0.06)
    cbar.set_label("amplitude")

    if title is not None:
        ax.set_title(title, pad=12)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=400, bbox_inches="tight", facecolor="white")

    #plt.show()
    plt.close(fig)



def christoffel_plane_cut(C, rho, plane="xz", ntheta=721):
    """
    Christoffel phase velocities in a coordinate plane.

    plane:
        "xy" : n = (cos(theta), sin(theta), 0)
        "xz" : n = (cos(theta), 0, sin(theta))
        "yz" : n = (0, cos(theta), sin(theta))

    Returns:
        theta  : angle array
        speeds : shape (ntheta, 3), sorted as [qS2, qS1, qP]
        dirs   : propagation directions
    """
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta)
    speeds = np.zeros((ntheta, 3))
    dirs = np.zeros((ntheta, 3))

    for i, th in enumerate(theta):
        if plane == "xy":
            n = np.array([np.cos(th), np.sin(th), 0.0])
        elif plane == "xz":
            n = np.array([np.cos(th), 0.0, np.sin(th)])
        elif plane == "yz":
            n = np.array([0.0, np.cos(th), np.sin(th)])
        else:
            raise ValueError("plane must be 'xy', 'xz', or 'yz'.")

        dirs[i] = n
        speeds[i] = wave_speeds_direction(C, rho, n)

    return theta, speeds, dirs


def plot_christoffel_plane_cut(
    C,
    rho,
    plane="xz",
    savepath=None,
    ntheta=1000,
    show_angle_plot=False,
):
    theta, speeds, _ = christoffel_plane_cut(C, rho, plane=plane, ntheta=ntheta)

    c_qS2 = speeds[:, 0]
    c_qS1 = speeds[:, 1]
    c_qP  = speeds[:, 2]

    color_qP  = "#D98C72"
    color_qS1 = "#B7E4C7"
    color_qS2 = "#FFAFCC"

    curves = [
        (c_qP,  r"$qP$",   2.6, color_qP),
        (c_qS1, r"$qS_1$", 2.2, color_qS1),
        (c_qS2, r"$qS_2$", 2.2, color_qS2),
    ]

    if plane == "xy":
        xlabel, ylabel = r"$x$", r"$y$"
        a = np.cos(theta)
        b = np.sin(theta)
    elif plane == "xz":
        xlabel, ylabel = r"$x$", r"$z$"
        a = np.cos(theta)
        b = np.sin(theta)
    elif plane == "yz":
        xlabel, ylabel = r"$y$", r"$z$"
        a = np.cos(theta)
        b = np.sin(theta)
    else:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'.")

    fig, ax = plt.subplots(figsize=(6.2, 6.2))

    for c, label, lw, color in curves:
        ax.plot(c * a, c * b, lw=lw, color=color, label=label)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(labelsize=12)
    ax.legend(frameon=False, fontsize=13, loc="upper right")

    rmax = np.max(speeds) * 1.08
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)

    fig.tight_layout()

    if savepath is None:
        savepath = f"christoffel_{plane}_cut.png"

    fig.savefig(savepath, dpi=400, bbox_inches="tight", facecolor="white")
    plt.show()

    if show_angle_plot:
        fig, ax = plt.subplots(figsize=(7.0, 3.8))
        ax.plot(theta, c_qP,  lw=2.4, color=color_qP,  label=r"$qP$")
        ax.plot(theta, c_qS1, lw=2.0, color=color_qS1, label=r"$qS_1$")
        ax.plot(theta, c_qS2, lw=2.0, color=color_qS2, label=r"$qS_2$")

        ax.set_xlabel(r"Propagation angle $\theta$", fontsize=14)
        ax.set_ylabel(r"Phase velocity", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.legend(frameon=False, fontsize=16, ncol=3)

        fig.tight_layout()
        fig.savefig(savepath.replace(".png", "_angle.png"),
                    dpi=400, bbox_inches="tight", facecolor="white")
        plt.show()

    return theta, speeds


# ------------------------------------------------------------
# Run simulation
# ------------------------------------------------------------

def run_simulation_single_cluster(
    N: int,
    L: float,
    T: float,
    rho: float,
    VP: float,
    VS: float,
    SCL: Bit3 = (0, 0, 0),
    source_width: float = 0.05,
    f0: float = 10.0,
    cfl: float = 1.0,
    medium: str = "vti",
    source_type: str = "shear_xz",
    C_mat=None,
):
    global DX, DY, DZ
    VCL = xor3(SCL, (1,1,1))
    Lx = Ly = Lz = L
    Nx = Ny = Nz = N

    DX = Lx / (Nx - 1)
    DY = Ly / (Ny - 1)
    DZ = Lz / (Nz - 1)

    #DEFINE STIFF MAT
    if C_mat is None:
        if medium == "isotropic":
            C = make_isotropic_C(rho, VP, VS)
        elif medium == "vti":
            C = make_vti_C(rho, VP, VS)
        else:
            raise ValueError(f"Unknown medium '{medium}'")
    else:
        C = C_mat

    if isinstance(C, dict):
        Cm = C[MATERIAL_GRIDS[0]]
    else:
        Cm = C


    #PLOT CHRISTOFFEL PLANAR CUTS
    plot_christoffel_plane_cut(C, rho, plane="xy", savepath="vti_christoffel_xy_cut.png")
    plot_christoffel_plane_cut(C, rho, plane="xz", savepath="vti_christoffel_xz_cut.png")
    plot_christoffel_plane_cut(C, rho, plane="yz", savepath="vti_christoffel_yz_cut.png")

    #TIME STEP
    vmax, direction, speeds = max_wave_speed(Cm, rho)
    dt = cfl * DX / (np.sqrt(3.0) * vmax)
    Nt = int(np.ceil(T / dt))
    dt = T / Nt

    V, S = build_state(SCL, Nx, Ny, Nz)

    if source_type == "shear_xz":
        Q = build_shear_xz_source(SCL, Nx, Ny, Nz, Lx, Ly, Lz, source_width)
    elif source_type == "explosive":
        Q = build_explosive_source(SCL, Nx, Ny, Nz, Lx, Ly, Lz, source_width)
    else:
        raise ValueError("source_type must be 'shear_xz' or 'explosive'.")

    t0 = 1 / f0

    snapshot_time = 0.45
    snapshot_n = int(snapshot_time / dt)

    ############# PPW ####################
    f_dom, f_max, freq, amp = estimate_fmax_from_source(
    dt=dt,
    Nt=Nt,
    f0=f0,
    t0=t0,
    cutoff=0.1
    )

    print("Dominant frequency:", f_dom)
    print("Maximum relevant frequency:", f_max)

    PPW_min = VS / (f_max * DX)
    print("Minimum PPW:", PPW_min)

    # plt.figure(figsize=(6, 4))
    # plt.plot(freq, amp)
    # plt.axvline(f_dom, linestyle="--", label=fr"$f_{{dom}}={f_dom:.2f}$")
    # plt.axvline(f_max, linestyle=":", label=fr"$f_{{max}}={f_max:.2f}$")
    # plt.xlabel("Frequency")
    # plt.ylabel("Amplitude")
    # plt.legend(frameon=False)
    # plt.tight_layout()
    # plt.show()
    ####################################
    for n in range(Nt):
        t = n * dt
        pulse = ricker(t, f0, t0)
        step_leapfrog(V, S, Q, C, SCL, rho, dt, pulse)

        if n == snapshot_n:
            vx = V[VX]
            vy = V[VY]
            vz = V[VZ]
            plot_three_planes_2d(vx, Lx, Ly, Lz, None, f"vtinumbastaggered_vxC111_t{t}N{N}.png")
            plot_three_planes_2d(vy, Lx, Ly, Lz, None, f"vtinumbastaggered_vyC111_t{t}N{N}.png")
            plot_three_planes_2d(vz, Lx, Ly, Lz, None, f"vtinumbastaggered_vzC111_t{t}N{N}.png")
            plot_lisitsa_pretty_fullplanes(
            vx,
            Lx, Ly, Lz,
            x0=Lx/2, y0=Ly/2, z0=Lz/2,
            title=None,
            savepath=f"vtinumbastagg3d_vx_fullplanes_t{t}_N{N}.png",
            upsample=2, show_plane_borders=True
            )
            plot_lisitsa_pretty_fullplanes(
            vy,
            Lx, Ly, Lz,
            x0=Lx/2, y0=Ly/2, z0=Lz/2,
            title=None,
            savepath=f"vtinumbastagg3d_vy_fullplaness_t{t}_N{N}.png",
            upsample=2, show_plane_borders=True
            )
            plot_lisitsa_pretty_fullplanes(
            vz,
            Lx, Ly, Lz,
            x0=Lx/2, y0=Ly/2, z0=Lz/2,
            title=None,
            savepath=f"vtinumbastagg3d_vz_fullplaness_t{t}_N{N}.png",
            upsample=2, show_plane_borders=True
            )
            break

    vx = V[VX]
    vy = V[VY]
    vz = V[VZ]

    sxx = S[XX]
    syy = S[YY]
    szz = S[ZZ]
    syz = S[YZ]
    sxz = S[XZ]
    sxy = S[XY]

   ###################################
    return {
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "dt": dt,
        "Nt": Nt,
        "SCL": SCL,
        "VCL": VCL,
    }


# ------------------------------------------------------------
# estimate fmax
# ------------------------------------------------------------

def estimate_fmax_from_source(dt, Nt, f0, t0, cutoff):
    t = np.arange(Nt) * dt
    s = ricker(t, f0, t0)

    F = np.fft.rfft(s)
    freq = np.fft.rfftfreq(Nt, d=dt)
    amp = np.abs(F)

    amp_max = np.max(amp)
    mask = amp >= cutoff * amp_max

    f_dom = freq[np.argmax(amp)]
    f_max = freq[mask][-1]

    return f_dom, f_max, freq, amp



# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    result = run_simulation_single_cluster(
    N = 301,
    L = 3,
    T = 1,
    rho = 2.5,
    VP = 3.0,
    VS = 1.0
    )
    
    print('Finished simulation')



if __name__ == '__main__':
    main()
