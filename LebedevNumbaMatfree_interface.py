"""
3D Elastic Wave Solver on a Lebedev grid with piecewise-homogeneous interfaces
==============================================================================

Purpose
-------
This script uses the binary grid/cluster notation developed in the report, to solve the 3D elastic wave 
equation in a piecewise-homogeneous medium with a horizontal interface using the Lebedev scheme. This
implementation uses 3D Numpy arrays and Numba-accelerated finite-difference kernels for the spatial derivative and explicit updates.


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
import matplotlib.pyplot as plt
from numba import njit, prange
import time


# ------------------------------------------------------------
# Global component conventions
# ------------------------------------------------------------


Bit3 = tuple[int, int, int] #defines cluster/grid state vector type


#shorthand notation
X, Y, Z = 0, 1, 2 
VX, VY, VZ = 0, 1, 2 
XX, YY, ZZ, YZ, XZ, XY = 0, 1, 2, 3, 4, 5

VCOMP_NAMES = ['x', 'y', 'z']
SCOMP_NAMES = ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']

#stress clusters and complementary velocity clusters
SCLUSTERS = [ #cluster-major ordering
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

#Component shifts 
V_SHIFT = [
    (1, 0, 0),  # vx
    (0, 1, 0),  # vy
    (0, 0, 1),  # vz
]

S_SHIFT = [
    (1, 1, 1),  # sigma_xx
    (1, 1, 1),  # sigma_yy
    (1, 1, 1),  # sigma_zz
    (1, 0, 0),  # sigma_yz
    (0, 1, 0),  # sigma_xz
    (0, 0, 1),  # sigma_xy
]

MATERIAL_GRIDS = [ #grid-major ordering
    (1, 1, 1),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
]


# ------------------------------------------------------------
# Binary cluster and grid logic
# ------------------------------------------------------------


def xor3(a: Bit3, b: Bit3): #Bitwise XOR
    return (a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2])


def toggle_axis(g: Bit3, axis: int):
    if axis == X:
        return (g[0] ^ 1, g[1], g[2])
    if axis == Y:
        return (g[0], g[1] ^ 1, g[2])
    if axis == Z:
        return (g[0], g[1], g[2] ^ 1)
    raise ValueError("axis must be 0, 1, or 2")


def shape_from_grid(g: Bit3, Nx: int, Ny: int, Nz: int): #returns shape of a given grid based on primary/dual grid components
    return (
        Nx if g[0] == 0 else Nx - 1,
        Ny if g[1] == 0 else Ny - 1,
        Nz if g[2] == 0 else Nz - 1,
    )


def velocity_grid(vcluster: Bit3, comp: int): #Calculates velocity grid from cluster idx and shift
    return xor3(vcluster, V_SHIFT[comp])


def stress_grid(scluster: Bit3, comp: int): #Calculates stress grid from cluster idx and shift
    return xor3(scluster, S_SHIFT[comp])


# ------------------------------------------------------------
# # 1D derivative kernels with nu,ma
# 0-bit axis = primary grid, length N
# 1-bit axis = dual grid, length N-1
# ------------------------------------------------------------

#to accelerate with numba, write as nested for loops, even though it could be written in array format with slicing

@njit(parallel=True, fastmath=False) #FUNCTION DECORATOR THAT TELLS NUMBA TO COMPILE TO FAST MACHINE CODE
def diff_x_p2d(u, h): #primary to dual grid in x
    nx, ny, nz = u.shape
    out = np.empty((nx - 1, ny, nz), dtype=u.dtype)
    ih = 1.0 / h
    for i in prange(nx - 1): #prange
        for j in range(ny):
            for k in range(nz):
                out[i, j, k] = (u[i + 1, j, k] - u[i, j, k]) * ih
    return out


@njit(parallel=True, fastmath=False)
def diff_y_p2d(u, h): #primary to dual grid in y
    nx, ny, nz = u.shape
    out = np.empty((nx, ny - 1, nz), dtype=u.dtype)
    ih = 1.0 / h
    for i in prange(nx):
        for j in range(ny - 1):
            for k in range(nz):
                out[i, j, k] = (u[i, j + 1, k] - u[i, j, k]) * ih
    return out


@njit(parallel=True, fastmath=False)
def diff_z_p2d(u, h): #primary to dual grid in z
    nx, ny, nz = u.shape
    out = np.empty((nx, ny, nz - 1), dtype=u.dtype)
    ih = 1.0 / h
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz - 1):
                out[i, j, k] = (u[i, j, k + 1] - u[i, j, k]) * ih
    return out


@njit(parallel=True, fastmath=False)
def diff_x_d2p(u, h):
    nx, ny, nz = u.shape
    out = np.empty((nx + 1, ny, nz), dtype=u.dtype)
    ih = 1.0 / h

    # D_d = -B.T / h
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
def diff_y_d2p(u, h):
    nx, ny, nz = u.shape
    out = np.empty((nx, ny + 1, nz), dtype=u.dtype)
    ih = 1.0 / h

    # D_d = -B.T / h
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
def diff_z_d2p(u, h):
    nx, ny, nz = u.shape
    out = np.empty((nx, ny, nz + 1), dtype=u.dtype)
    ih = 1.0 / h

    # D_d = -B.T / h
    for i in prange(nx):
        for j in range(ny):
            out[i, j, 0] = u[i, j, 0] * ih
            out[i, j, nz] = -u[i, j, nz - 1] * ih

    for i in prange(nx):
        for j in range(ny):
            for k in range(1, nz):
                out[i, j, k] = (u[i, j, k] - u[i, j, k - 1]) * ih

    return out


def diff_axis(u: np.ndarray, grid: Bit3, axis: int, h: float) -> tuple[np.ndarray, Bit3]: #this function differentiates a field along an axis
 
    bit = grid[axis]

    if axis == X:
        out = diff_x_p2d(u, h) if bit == 0 else diff_x_d2p(u, h)
    elif axis == Y:
        out = diff_y_p2d(u, h) if bit == 0 else diff_y_d2p(u, h)
    elif axis == Z:
        out = diff_z_p2d(u, h) if bit == 0 else diff_z_d2p(u, h)
    else:
        raise ValueError("axis must be X, Y, or Z")

    return out, toggle_axis(grid, axis) #returns differentiated array and its new grid


# ------------------------------------------------------------
# Small array-algebra kernels
# ------------------------------------------------------------


@njit(parallel=True, fastmath=False)
def add2(a, b): #addition of 2 arrays made for numba
    nx, ny, nz = a.shape
    out = np.empty_like(a)
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                out[i, j, k] = a[i, j, k] + b[i, j, k]
    return out


@njit(parallel=True, fastmath=False)
def add3(a, b, c): #addition of 3 arrays made for numba
    nx, ny, nz = a.shape
    out = np.empty_like(a)
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                out[i, j, k] = a[i, j, k] + b[i, j, k] + c[i, j, k]
    return out


@njit(parallel=True, fastmath=False)
def axpy_inplace(y, a, x): #equivalent to the numpy expression y+= a*x
    nx, ny, nz = y.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                y[i, j, k] += a * x[i, j, k]


@njit(parallel=True, fastmath=False)
def apply_C_grid(exx, eyy, ezz, eyz, exz, exy, C): 
    """
    Apply 6x6 stiffness matrix C pointwise on one material grid.
    All six strain-rate components must have the same shape.
    """
    nx, ny, nz = exx.shape

    sxx = np.empty_like(exx)
    syy = np.empty_like(exx)
    szz = np.empty_like(exx)
    syz = np.empty_like(exx)
    sxz = np.empty_like(exx)
    sxy = np.empty_like(exx)

    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                a0 = exx[i, j, k]
                a1 = eyy[i, j, k]
                a2 = ezz[i, j, k]
                a3 = eyz[i, j, k]
                a4 = exz[i, j, k]
                a5 = exy[i, j, k]

                sxx[i, j, k] = (
                    C[0, 0]*a0 + C[0, 1]*a1 + C[0, 2]*a2
                    + C[0, 3]*a3 + C[0, 4]*a4 + C[0, 5]*a5
                )
                syy[i, j, k] = (
                    C[1, 0]*a0 + C[1, 1]*a1 + C[1, 2]*a2
                    + C[1, 3]*a3 + C[1, 4]*a4 + C[1, 5]*a5
                )
                szz[i, j, k] = (
                    C[2, 0]*a0 + C[2, 1]*a1 + C[2, 2]*a2
                    + C[2, 3]*a3 + C[2, 4]*a4 + C[2, 5]*a5
                )
                syz[i, j, k] = (
                    C[3, 0]*a0 + C[3, 1]*a1 + C[3, 2]*a2
                    + C[3, 3]*a3 + C[3, 4]*a4 + C[3, 5]*a5
                )
                sxz[i, j, k] = (
                    C[4, 0]*a0 + C[4, 1]*a1 + C[4, 2]*a2
                    + C[4, 3]*a3 + C[4, 4]*a4 + C[4, 5]*a5
                )
                sxy[i, j, k] = (
                    C[5, 0]*a0 + C[5, 1]*a1 + C[5, 2]*a2
                    + C[5, 3]*a3 + C[5, 4]*a4 + C[5, 5]*a5
                )

    return sxx, syy, szz, syz, sxz, sxy


# ------------------------------------------------------------
# State construction
# ------------------------------------------------------------

def zeros_on_grid(grid: Bit3, Nx: int, Ny: int, Nz: int) -> np.ndarray: 
    return np.zeros(shape_from_grid(grid, Nx, Ny, Nz), dtype=np.float64)


def build_state(Nx: int, Ny: int, Nz: int):
    """
    Returns V, S as nested Python lists.

    V[cluster_id][component]
    S[cluster_id][component]
    """
    V = []
    S = []

    for ci in range(4):
        vcluster = VCLUSTERS[ci]
        scluster = SCLUSTERS[ci]

        Vci = []
        for comp in range(3):
            g = velocity_grid(vcluster, comp)
            Vci.append(zeros_on_grid(g, Nx, Ny, Nz))
        V.append(Vci)

        Sci = []
        for comp in range(6):
            g = stress_grid(scluster, comp)
            Sci.append(zeros_on_grid(g, Nx, Ny, Nz))
        S.append(Sci)

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

def gaussian_on_grid(
    grid: Bit3,
    Nx: int, Ny: int, Nz: int,
    Lx: float, Ly: float, Lz: float,
    x0: float, y0: float, z0: float,
    width: float,
):
    x, y, z = coords_for_grid(grid, Nx, Ny, Nz, Lx, Ly, Lz)
    Xg, Yg, Zg = np.meshgrid(x, y, z, indexing="ij")
    return np.exp(-((Xg - x0)**2 + (Yg - y0)**2 + (Zg - z0)**2) / (2.0 * width**2))


def build_explosive_source(
    Nx: int, Ny: int, Nz: int,
    Lx: float, Ly: float, Lz: float,
    width: float, x0 = None, y0 = None, z0 = None
    ):
    """
    builds a pressure-like source on sigma_xx, sigma_yy, sigma_zz
    for each stress cluster
    """
    if x0 is None:
        x0 = Lx / 2.0
    if y0 is None:
        y0 = Ly / 2
    if z0 is None:
        z0 = Lz / 2
    
    #x0, y0, z0 = Lx / 2.0, Ly / 2.0, Lz / 2.0

    Q = []
    for ci in range(4):
        scluster = SCLUSTERS[ci]
        Qci = []

        for comp in range(6):
            g = stress_grid(scluster, comp)
            arr = np.zeros(shape_from_grid(g, Nx, Ny, Nz), dtype=np.float64)

                   
            if comp in (XX, YY, ZZ):
                arr[:] = gaussian_on_grid(g, Nx, Ny, Nz, Lx, Ly, Lz, x0, y0, z0, width)

            Qci.append(arr)

        Q.append(Qci)

    return Q

def build_shear_source(
    Nx: int, Ny: int, Nz: int,
    Lx: float, Ly: float, Lz: float,
    width: float, x0 = None, y0 = None, z0 = None
    ):
    """
    builds a xz-shear stress source for each stress cluster
    """
    if x0 is None:
       x0 = Lx / 2.0
    if y0 is None:
        y0 = Ly / 2
    if z0 is None:
        z0 = Lz / 2
    #x0, y0, z0 = Lx / 2.0, Ly / 2.0, Lz / 2.0

    Q = []
    for ci in range(4):
        scluster = SCLUSTERS[ci]
        Qci = []

        for comp in range(6):
            g = stress_grid(scluster, comp)
            arr = np.zeros(shape_from_grid(g, Nx, Ny, Nz), dtype=np.float64)

            #only excite gaussian for sigma_xz
            if comp == XZ:  
                arr[:] = gaussian_on_grid(g, Nx, Ny, Nz, Lx, Ly, Lz, x0, y0, z0, width)  
           
            Qci.append(arr)

        Q.append(Qci)

    return Q


def ricker(t: float, f0: float, t0: float) -> float:
    a = np.pi * f0 * (t - t0)
    return (1.0 - 2.0 * a*a) * np.exp(-a*a)


# ------------------------------------------------------------
# Interface
# ------------------------------------------------------------

def safe_z_interface(k: int, dz: float, offset: float = 0.75):
    """
    Primary z points: z_i = i dz
    Dual z points:    zhat_i = (i + 0.5) dz

    offset=0.75 puts the interface between zhat_k and z_{k+1}
    """
    if offset in (0.0, 0.5, 1.0):
        raise ValueError("Interface should not be placed directly on a primary or dual grid point.")

    return (k + offset) * dz


def build_layer_id_by_grid_z_interface(
    Nx: int,
    Ny: int,
    Nz: int,
    Lx: float,
    Ly: float,
    Lz: float,
    z_interface: float,
    ):
    """
    Build one 1D layer-id array for each Lebedev material grid

    layer_id_by_grid[g][k] = 0  -> use C_below
    layer_id_by_grid[g][k] = 1  -> use C_above

    Avoiding storing Cg with shape (nx, ny, nz, 6, 6)
    """
    layer_id_by_grid = {}

    for g in MATERIAL_GRIDS:
        _, _, z = coords_for_grid(g, Nx, Ny, Nz, Lx, Ly, Lz)

        layer_id = np.zeros(len(z), dtype=np.uint8)
        layer_id[z >= z_interface] = 1

        layer_id_by_grid[g] = layer_id

    return layer_id_by_grid

# def build_C_by_grid_z_interface(
    #     Nx: int,
    #     Ny: int,
    #     Nz: int,
    #     Lx: float,
    #     Ly: float,
    #     Lz: float,
    #     z_interface: float,
    #     C_below: np.ndarray,
    #     C_above: np.ndarray,
    # ):
    # """
    # Build spatially varying stiffness arrays on each Lebedev material grid.

    # Returns:
    #     C_by_grid[g] with shape shape_from_grid(g) + (6, 6)

    # The rule is:
    #     z < z_interface  -> C_below
    #     z >= z_interface -> C_above
    # """
    # C_below = np.asarray(C_below, dtype=np.float64)
    # C_above = np.asarray(C_above, dtype=np.float64)

    # if C_below.shape != (6, 6):
    #     raise ValueError("C_below must have shape (6, 6)")
    # if C_above.shape != (6, 6):
    #     raise ValueError("C_above must have shape (6, 6)")

    # C_by_grid = {}

    # for g in MATERIAL_GRIDS:
    #     x, y, z = coords_for_grid(g, Nx, Ny, Nz, Lx, Ly, Lz)
    #     nx, ny, nz = len(x), len(y), len(z)

    #     Cg = np.empty((nx, ny, nz, 6, 6), dtype=np.float64)

    #     for k in range(nz):
    #         if z[k] < z_interface:
    #             Cg[:, :, k, :, :] = C_below
    #         else:
    #             Cg[:, :, k, :, :] = C_above

    #     C_by_grid[g] = Cg

    # return C_by_grid



@njit(parallel=True, fastmath=False)
def apply_C_grid_two_layer(
    exx, eyy, ezz, eyz, exz, exy,
    C_below,
    C_above,
    layer_id,
    ):
    """
    Apply a two-layer stiffness tensor on one Lebedev material grid

    layer_id[k] = 0 -> C_below
    layer_id[k] = 1 -> C_above

    This is memory-light because it does not store C(i,j,k,:,:)
    """
    nx, ny, nz = exx.shape

    sxx = np.empty_like(exx)
    syy = np.empty_like(exx)
    szz = np.empty_like(exx)
    syz = np.empty_like(exx)
    sxz = np.empty_like(exx)
    sxy = np.empty_like(exx)

    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):

                if layer_id[k] == 0:
                    C = C_below
                else:
                    C = C_above

                a0 = exx[i, j, k]
                a1 = eyy[i, j, k]
                a2 = ezz[i, j, k]
                a3 = eyz[i, j, k]
                a4 = exz[i, j, k]
                a5 = exy[i, j, k]

                sxx[i, j, k] = (
                    C[0, 0]*a0 + C[0, 1]*a1 + C[0, 2]*a2
                    + C[0, 3]*a3 + C[0, 4]*a4 + C[0, 5]*a5
                )

                syy[i, j, k] = (
                    C[1, 0]*a0 + C[1, 1]*a1 + C[1, 2]*a2
                    + C[1, 3]*a3 + C[1, 4]*a4 + C[1, 5]*a5
                )

                szz[i, j, k] = (
                    C[2, 0]*a0 + C[2, 1]*a1 + C[2, 2]*a2
                    + C[2, 3]*a3 + C[2, 4]*a4 + C[2, 5]*a5
                )

                syz[i, j, k] = (
                    C[3, 0]*a0 + C[3, 1]*a1 + C[3, 2]*a2
                    + C[3, 3]*a3 + C[3, 4]*a4 + C[3, 5]*a5
                )

                sxz[i, j, k] = (
                    C[4, 0]*a0 + C[4, 1]*a1 + C[4, 2]*a2
                    + C[4, 3]*a3 + C[4, 4]*a4 + C[4, 5]*a5
                )

                sxy[i, j, k] = (
                    C[5, 0]*a0 + C[5, 1]*a1 + C[5, 2]*a2
                    + C[5, 3]*a3 + C[5, 4]*a4 + C[5, 5]*a5
                )

    return sxx, syy, szz, syz, sxz, sxy


def apply_material_grid_major_two_layer(
    strain_by_grid,
    C_below: np.ndarray,
    C_above: np.ndarray,
    layer_id_by_grid,
):
    """
    Applies sigma_rhs = C(z) * strain_rhs on each Lebedev material grid

    Uses grid-major strain storage, but avoids full C_by_grid storage
    """
    C_below = np.asarray(C_below, dtype=np.float64)
    C_above = np.asarray(C_above, dtype=np.float64)

    if C_below.shape != (6, 6):
        raise ValueError("C_below must have shape (6, 6)")
    if C_above.shape != (6, 6):
        raise ValueError("C_above must have shape (6, 6)")

    stress_rhs_by_grid = {}

    for g in MATERIAL_GRIDS:
        exx, eyy, ezz, eyz, exz, exy = strain_by_grid[g]
        layer_id = layer_id_by_grid[g]

        stress_rhs_by_grid[g] = apply_C_grid_two_layer(
            exx, eyy, ezz, eyz, exz, exy,
            C_below,
            C_above,
            layer_id,
        )

    return stress_rhs_by_grid


def compute_strain_grid_major(V):
    """
    Computes D_v v and stores the six strain-rate components by material grid

    Returns:
        strain_by_grid[g] = [exx, eyy, ezz, eyz, exz, exy]
    """
    strain_by_grid = {g: [None, None, None, None, None, None] for g in MATERIAL_GRIDS}

    for ci in range(4):
        vcluster = VCLUSTERS[ci]

        vx = V[ci][VX]
        vy = V[ci][VY]
        vz = V[ci][VZ]

        gvx = velocity_grid(vcluster, VX)
        gvy = velocity_grid(vcluster, VY)
        gvz = velocity_grid(vcluster, VZ)

        # Normal strain rates
        exx, g_exx = diff_axis(vx, gvx, X, DX)
        eyy, g_eyy = diff_axis(vy, gvy, Y, DY)
        ezz, g_ezz = diff_axis(vz, gvz, Z, DZ)

        strain_by_grid[g_exx][XX] = exx
        strain_by_grid[g_eyy][YY] = eyy
        strain_by_grid[g_ezz][ZZ] = ezz

        # Engineering shear strain rates
        dvy_dz, g1 = diff_axis(vy, gvy, Z, DZ)
        dvz_dy, g2 = diff_axis(vz, gvz, Y, DY)
        if g1 != g2:
            raise RuntimeError("Grid mismatch in eyz")
        strain_by_grid[g1][YZ] = add2(dvy_dz, dvz_dy)

        dvx_dz, g1 = diff_axis(vx, gvx, Z, DZ)
        dvz_dx, g2 = diff_axis(vz, gvz, X, DX)
        if g1 != g2:
            raise RuntimeError("Grid mismatch in exz")
        strain_by_grid[g1][XZ] = add2(dvx_dz, dvz_dx)

        dvx_dy, g1 = diff_axis(vx, gvx, Y, DY)
        dvy_dx, g2 = diff_axis(vy, gvy, X, DX)
        if g1 != g2:
            raise RuntimeError("Grid mismatch in exy")
        strain_by_grid[g1][XY] = add2(dvx_dy, dvy_dx)

    for g in MATERIAL_GRIDS:
        if any(x is None for x in strain_by_grid[g]):
            raise RuntimeError(f"Incomplete strain components on grid {g}")

    return strain_by_grid



def update_stresses(S, stress_rhs_by_grid, Q, dt: float, pulse: float):
    """
    Scatter grid-major stress RHS back to stress-cluster storage
    """
    for ci in range(4):
        scluster = SCLUSTERS[ci]

        for comp in range(6):
            g = stress_grid(scluster, comp)
            rhs = stress_rhs_by_grid[g][comp]
            axpy_inplace(S[ci][comp], dt, rhs) #equiv to S[ci][comp] += dt*rhs

            if Q is not None:
                axpy_inplace(S[ci][comp], dt * pulse, Q[ci][comp])


def update_velocities(V, S, rho: float, dt: float):
    """
    Computes D_sigma sigma and updates velocity fields
    """
    scale = dt / rho

    for ci in range(4):
        scluster = SCLUSTERS[ci]
        vcluster = VCLUSTERS[ci]

        sxx = S[ci][XX]
        syy = S[ci][YY]
        szz = S[ci][ZZ]
        syz = S[ci][YZ]
        sxz = S[ci][XZ]
        sxy = S[ci][XY]

        gsxx = stress_grid(scluster, XX)
        gsyy = stress_grid(scluster, YY)
        gszz = stress_grid(scluster, ZZ)
        gsyz = stress_grid(scluster, YZ)
        gsxz = stress_grid(scluster, XZ)
        gsxy = stress_grid(scluster, XY)

        # vx equation: d_x sxx + d_y sxy + d_z sxz
        a, ga = diff_axis(sxx, gsxx, X, DX)
        b, gb = diff_axis(sxy, gsxy, Y, DY)
        c, gc = diff_axis(sxz, gsxz, Z, DZ)
        gvx = velocity_grid(vcluster, VX)
        if not (ga == gb == gc == gvx):
            raise RuntimeError(f"Grid mismatch in vx update: {ga}, {gb}, {gc}, expected {gvx}")
        rhs_vx = add3(a, b, c)
        axpy_inplace(V[ci][VX], scale, rhs_vx)

        # vy equation: d_x sxy + d_y syy + d_z syz
        a, ga = diff_axis(sxy, gsxy, X, DX)
        b, gb = diff_axis(syy, gsyy, Y, DY)
        c, gc = diff_axis(syz, gsyz, Z, DZ)
        gvy = velocity_grid(vcluster, VY)
        if not (ga == gb == gc == gvy):
            raise RuntimeError(f"Grid mismatch in vy update: {ga}, {gb}, {gc}, expected {gvy}")
        rhs_vy = add3(a, b, c)
        axpy_inplace(V[ci][VY], scale, rhs_vy)

        # vz equation: d_x sxz + d_y syz + d_z szz
        a, ga = diff_axis(sxz, gsxz, X, DX)
        b, gb = diff_axis(syz, gsyz, Y, DY)
        c, gc = diff_axis(szz, gszz, Z, DZ)
        gvz = velocity_grid(vcluster, VZ)
        if not (ga == gb == gc == gvz):
            raise RuntimeError(f"Grid mismatch in vz update: {ga}, {gb}, {gc}, expected {gvz}")
        rhs_vz = add3(a, b, c)
        axpy_inplace(V[ci][VZ], scale, rhs_vz)





def add_source_to_strain_grid_major(strain_by_grid, Q, pulse):
    if Q is None:
        return

    for ci in range(4):
        scluster = SCLUSTERS[ci]

        for comp in range(6):
            g = stress_grid(scluster, comp)

            # This assumes Q[ci][comp] lives on the same stress/material grid.
            if Q[ci][comp] is not None:
                axpy_inplace(strain_by_grid[g][comp], pulse, Q[ci][comp])



# ------------------------------------------------------------
# One leapfrog step
# ------------------------------------------------------------


def step_leapfrog_two_layer(
    V,
    S,
    Q,
    C_below: np.ndarray,
    C_above: np.ndarray,
    layer_id_by_grid,
    rho: float,
    dt: float,
    pulse: float,
    ):
    strain_rhs = compute_strain_grid_major(V)

    add_source_to_strain_grid_major(strain_rhs, Q, pulse)

    stress_rhs = apply_material_grid_major_two_layer(
        strain_rhs,
        C_below,
        C_above,
        layer_id_by_grid,
    )

    update_stresses(S, stress_rhs, None, dt, pulse)

    update_velocities(V, S, rho, dt)




def total_l2(V, S): #total norm 
    val = 0.0
    for ci in range(4):
        for a in V[ci]:
            val += np.sum(a*a)
        for a in S[ci]:
            val += np.sum(a*a)
    return np.sqrt(val)


# ------------------------------------------------------------
# Stiffness matrices
# ------------------------------------------------------------

def make_vti_C(rho: float, VP: float, VS: float):
    epsilon = 0.334
    gamma = 0.575
    delta_star = 0.93

    C33 = rho * VP**2
    C44 = rho * VS**2

    C11 = C33 * (1.0 + 2.0 * epsilon)
    C66 = C44 * (1.0 + 2.0 * gamma)
    C12 = C11 - 2.0 * C66
    C13 = C33 * delta_star - C44

    C = np.array([
        [C11, C12, C13, 0.0, 0.0, 0.0],
        [C12, C11, C13, 0.0, 0.0, 0.0],
        [C13, C13, C33, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, C44, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, C44, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, C66],
    ], dtype=np.float64)

    return C


def make_isotropic_C(rho: float, VP: float, VS: float):
    mu = rho * VS**2
    lam = rho * (VP**2 - 2.0 * VS**2)

    C11 = lam + 2.0 * mu
    C12 = lam
    C44 = mu

    C = np.array([
        [C11, C12, C12, 0.0, 0.0, 0.0],
        [C12, C11, C12, 0.0, 0.0, 0.0],
        [C12, C12, C11, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, C44, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, C44, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, C44],
    ], dtype=np.float64)

    return C

def make_synthetic_anisotropic_C(rho: float, VP: float, VS: float):
    C = make_vti_C(rho, VP, VS)

    #add normal-shear coupling terms
    C[0, 4] = C[4, 0] = 4.0   # C15
    C[2, 4] = C[4, 2] = 1.2   # C35
    C[1, 3] = C[3, 1] = 1.8   # C24
    C[0, 5] = C[5, 0] = 4.6   # C16

    #check positive definiteness.
    eigs = np.linalg.eigvalsh(C)
    if np.min(eigs) <= 0:
        raise ValueError(f"C is not positive definite. Eigenvalues: {eigs}")

    return C



# ------------------------------------------------------------
# create TTI medium
# ------------------------------------------------------------

VOIGT_PAIRS = [
    (0, 0),  # xx
    (1, 1),  # yy
    (2, 2),  # zz
    (1, 2),  # yz
    (0, 2),  # xz
    (0, 1),  # xy
]


def voigt_to_tensor(C_voigt: np.ndarray) -> np.ndarray:
    """
    Convert 6x6 Voigt stiffness matrix to 4th-order tensor.
    Assumes Voigt ordering [xx, yy, zz, yz, xz, xy]
    """
    C4 = np.zeros((3, 3, 3, 3), dtype=np.float64)

    for I, (i, j) in enumerate(VOIGT_PAIRS):
        for J, (k, l) in enumerate(VOIGT_PAIRS):
            val = C_voigt[I, J]

            C4[i, j, k, l] = val
            C4[j, i, k, l] = val
            C4[i, j, l, k] = val
            C4[j, i, l, k] = val

    return C4


def tensor_to_voigt(C4: np.ndarray) -> np.ndarray:
    """
    Convert 4th-order stiffness tensor to 6x6 Voigt matrix
    """
    C_voigt = np.zeros((6, 6), dtype=np.float64)

    for I, (i, j) in enumerate(VOIGT_PAIRS):
        for J, (k, l) in enumerate(VOIGT_PAIRS):
            C_voigt[I, J] = C4[i, j, k, l]

    return 0.5 * (C_voigt + C_voigt.T)


def rotation_y(alpha_deg: float) -> np.ndarray:
    """
    Rotation matrix in y-dir
    [[ cos a, 0, -sin a],
     [ 0,     1,  0    ],
     [ sin a, 0,  cos a]]
    """
    a = np.deg2rad(alpha_deg)
    c = np.cos(a)
    s = np.sin(a)

    return np.array([
        [ c, 0.0, -s],
        [0.0, 1.0, 0.0],
        [ s, 0.0,  c],
    ], dtype=np.float64)


def rotate_stiffness_y(C_voigt: np.ndarray, alpha_deg: float) -> np.ndarray:
    """
    Rotate stiffness tensor by alpha_deg about y-axis
    """
    A = rotation_y(alpha_deg)
    C4 = voigt_to_tensor(C_voigt)

    C4_rot = np.einsum(
        "ip,jq,kr,ls,pqrs->ijkl",
        A, A, A, A,
        C4
    )

    C_rot = tensor_to_voigt(C4_rot)

    eigs = np.linalg.eigvalsh(C_rot)
    if np.min(eigs) <= 0.0:
        raise ValueError(f"Rotated stiffness is not positive definite. Eigenvalues: {eigs}")

    return C_rot


def make_tti_C(
    rho: float,
    VP: float,
    VS: float,
    epsilon: float,
    gamma: float,
    delta: float,
    tilt_deg: float):
    
    C_vti = make_vti_C(rho=rho, VP=VP, VS=VS)#, epsilon=epsilon, gamma=gamma, delta=delta)

    C_tti = rotate_stiffness_y(C_vti, tilt_deg)
    return C_tti



# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def plot_three_planes_2d(field3: np.ndarray, Lx: float, Ly: float, Lz: float,  z_int: int, title: str, savepath: str | None = None) -> None:
    nx, ny, nz = field3.shape
    ix, iy, iz = nx // 2, ny // 2, nz // 2

    xy = field3[:, :, iz]
    xz = field3[:, iy, :]
    yz = field3[ix, :, :]

    vmax = max(np.max(np.abs(xy)), np.max(np.abs(xz)), np.max(np.abs(yz)))
    if vmax == 0.0:
        vmax = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    im = axes[0].imshow(xy.T, origin="lower", extent=[0, Lx, 0, Ly], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("xy", fontsize=18)
    axes[0].set_xlabel("x", fontsize=18)
    axes[0].set_ylabel("y", fontsize=18)

   
    axes[1].imshow(xz.T, origin="lower", extent=[0, Lx, 0, Lz], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].axhline(z_int, color="k", ls="--", lw=1.2)
    axes[1].set_title("xz", fontsize=18)
    axes[1].set_xlabel("x", fontsize=18)
    axes[1].set_ylabel("z", fontsize=18)

    axes[2].imshow(yz.T, origin="lower", extent=[0, Ly, 0, Lz], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[2].axhline(z_int, color="k", ls="--", lw=1.2)
    axes[2].set_title("yz", fontsize=18)
    axes[2].set_xlabel("y", fontsize=18)
    axes[2].set_ylabel("z", fontsize=18)

    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle(title)

    if savepath is not None:
        plt.savefig(savepath, dpi=250, bbox_inches="tight")
    plt.close(fig)
    #plt.show()


def plot_lisitsa_pretty_fullplanes(
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

    ax.set_xlabel("x", fontsize=18)
    ax.set_ylabel("y", fontsize=18)
    ax.set_zlabel("z", fontsize=18)

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


# ------------------------------------------------------------
# Christoffel curves and max wave speeds
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

def plot_christoffel_interface_comparison(
    C_below,
    C_above,
    rho,
    plane="xz",
    ntheta=1000,
    savepath=None,
):
    """
    Plot Christoffel phase-velocity curves for the two media
    separated by a stiffness interface.

    Solid curves  : below-interface medium
    Dashed curves : above-interface medium
    """
    import numpy as np
    import matplotlib.pyplot as plt

    theta_b, speeds_b, _ = christoffel_plane_cut(
        C_below, rho, plane=plane, ntheta=ntheta
    )
    theta_a, speeds_a, _ = christoffel_plane_cut(
        C_above, rho, plane=plane, ntheta=ntheta
    )

    if plane == "xy":
        xlabel, ylabel = r"$x$", r"$y$"
        xdir = np.cos(theta_b)
        ydir = np.sin(theta_b)
    elif plane == "xz":
        xlabel, ylabel = r"$x$", r"$z$"
        xdir = np.cos(theta_b)
        ydir = np.sin(theta_b)
    elif plane == "yz":
        xlabel, ylabel = r"$y$", r"$z$"
        xdir = np.cos(theta_b)
        ydir = np.sin(theta_b)
    else:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'.")

    labels = [r"$qS_2$", r"$qS_1$", r"$qP$"]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    for m in range(3):
        # below interface: solid
        ax.plot(
            speeds_b[:, m] * xdir,
            speeds_b[:, m] * ydir,
            lw=2.2,
            label=labels[m] + " below",
        )

        # above interface: dashed
        ax.plot(
            speeds_a[:, m] * xdir,
            speeds_a[:, m] * ydir,
            lw=2.2,
            ls="--",
            label=labels[m] + " above",
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(f"Christoffel phase-velocity curves, {plane}-plane")

    rmax = 1.08 * max(np.max(speeds_b), np.max(speeds_a))
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)

    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=10, ncol=2)
    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=400, bbox_inches="tight", facecolor="white")

    plt.show()

    return fig, ax

# ------------------------------------------------------------
# Run Lebedev simulation with interface
# ------------------------------------------------------------


def run_Lebedev_simulation(
    N: int,
    L: float,
    T: float,
    rho: float,
    VP_above: float,
    VP_below: float,
    VS_above: float,
    VS_below: float,
    medium_above: str,
    medium_below: str,
    sclusters: list = SCLUSTERS,
    vclusters: list = VCLUSTERS,
    source_width: float = 0.05,
    f0: float = 10.0,
    cfl: float = 1.0,
    source_type: str = 'explosive'
    ):

    global DX, DY, DZ
    Lx = Ly = Lz = L
    Nx = Ny = Nz = N
    

    ######## STEP SIZES ########
    DX = Lx / (Nx - 1)
    DY = Ly / (Ny - 1)
    DZ = Lz / (Nz - 1)
    
    epsilon = 0.334
    gamma = 0.575
    delta = 0.93
    ######## MEDIUM BELOW ########
    if medium_below == "ISO":
        C_below = make_isotropic_C(rho, VP_below, VS_below)
    elif medium_below == "VTI":
        C_below = make_vti_C(rho, VP_below, VS_below)
    elif medium_below == 'TTI':
        C_below = make_tti_C(rho, VP_below, VS_below, epsilon, gamma, delta, 45)
    elif medium_below == "ANI":
        C_below = make_synthetic_anisotropic_C(rho, VP_below, VS_below)
    else:
        raise ValueError(f"Unknown medium '{medium_below}'")

    ######## MEDIUM ABOVE ########
    if medium_above == "ISO":
        C_above= make_isotropic_C(rho, VP_above, VS_above)
        C_above = make_isotropic_C(rho, VP_above, VS_above)
    elif medium_above == "VTI":
        C_above = make_vti_C(rho, VP_above, VS_above)
    elif medium_above == 'TTI':
        C_above = make_tti_C(rho, VP_above, VS_above, epsilon, gamma, delta, 45)
    elif medium_above == "ANI":
        C_above = make_synthetic_anisotropic_C(rho, VP_above, VS_above)
    else:
        raise ValueError(f"Unknown medium '{medium_above}'")


    plot_christoffel_interface_comparison(
    C_below,
    C_above,
    rho,
    plane="xz",
    savepath="christoffel_interface_xz.png",
    )
    plot_christoffel_interface_comparison(
    C_below,
    C_above,
    rho,
    plane="yz",
    savepath="christoffel_interface_yz.png",
    )
    plot_christoffel_interface_comparison(
    C_below,
    C_above,
    rho,
    plane="xy",
    savepath="christoffel_interface_xy.png",
    )
    
    

    # PLOT CHRISTOFFEL CUTS 
    # plot_christoffel_plane_cut(C, rho, plane="xy", savepath=f"{medium}_christoffel_xy_cut.png")
    # plot_christoffel_plane_cut(C, rho, plane="xz", savepath=f"{medium}_christoffel_xz_cut.png")
    # plot_christoffel_plane_cut(C, rho, plane="yz", savepath=f"{medium}_christoffel_yz_cut.png")


    print("C_below eigenvalues:", np.linalg.eigvalsh(C_below))
    print("C_above eigenvalues:", np.linalg.eigvalsh(C_above))
    

    # TIMESTEP

    vmax_below, direction_below, speeds_below = max_wave_speed(C_below, rho)
    vmax_above, direction_above, speeds_above = max_wave_speed(C_above, rho)
    vmax = max(vmax_above, vmax_below)
    print(f'vmax below: {vmax_below}')
    print(f'vmax above: {vmax_above}')
    print(f'used vmax={vmax}')


    cfl = 1
    dt = cfl * DX / (np.sqrt(3.0) * vmax)
    T = 1.0
    Nt = int(np.ceil(T / dt))

    #target_interface = 1.20
    #k_interface = int(np.floor(target_interface / DZ - 0.75))#Nz // 2
    k_interface = Nz // 2
    z_interface = safe_z_interface(k_interface, DZ, offset = 0.75)
    print("z_interface =", z_interface)
    print("interface between:")
    print("dual point    =", (k_interface + 0.5) * DZ)
    print("primary point =", (k_interface + 1.0) * DZ)



    layer_id_by_grid = build_layer_id_by_grid_z_interface(
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        z_interface=z_interface,
    )


    ###### START TIMER ######
    start = time.perf_counter()

    print(f"dt = {dt:.6e}, Nt = {Nt}, h = {DX:.6e}")
    z0 = z_interface - 5 * source_width
    print(f'source z0 = {z0}')
    print(f'distance to interface = {z_interface - z0}')
    # Build state and source
    V, S = build_state(Nx, Ny, Nz)
    if source_type == "shear":
        Q = build_shear_source(Nx, Ny, Nz, Lx, Ly, Lz, source_width, None, None, z0)
    elif source_type == "explosive":
        Q = build_explosive_source(Nx, Ny, Nz, Lx, Ly, Lz, source_width, None, None, z0)
    elif source_type is None:
        Q = None
    else:
        raise ValueError("source_type must be 'shear_xz' or 'explosive'.")




    t_plot = 0.35#1.5 / vmax
    print(f't_plot={t_plot}')
    t0 = 1.0 / f0
    snapshot_time = t_plot
    snapshot_n = int(snapshot_time / dt)
    for n in range(Nt):
        t = n * dt
        pulse = ricker(t, f0, t0)
        step_leapfrog_two_layer(
            V,
            S,
            Q,
            C_below,
            C_above,
            layer_id_by_grid,
            rho,
            dt,
            pulse,
        )
      

        if n % 50 == 0:
            print(f"n={n:5d}/{Nt}, t={t:.4f}, norm={total_l2(V, S):.6e}")

        if n == snapshot_n:
            vx_copies = []
            vy_copies = []
            vz_copies = []
            for ci in range(4):
                vcluster = VCLUSTERS[ci]   
                vx_copies.append(V[ci][VX])
                vy_copies.append(V[ci][VY])
                vz_copies.append(V[ci][VZ])
            

            vx_c111 = vx_copies[0]
            vx_c100 = vx_copies[1]
            vx_c010 = vx_copies[2]
            vx_c001 = vx_copies[3]


            vy_c111 = vy_copies[0]
            vy_c100 = vy_copies[1]
            vy_c010 = vy_copies[2]
            vy_c001 = vy_copies[3]

            
            vz_c111 = vz_copies[0]
            vz_c100 = vz_copies[1]
            vz_c010 = vz_copies[2]
            vz_c001 = vz_copies[3]

            # plot_three_planes_2d(vx_c111, Lx, Ly, Lz, z_interface, None, f"above{medium_above}below{medium_below}numbaLebedev_inteface_vxC111_t{t}N{N}.png")
            # plot_three_planes_2d(vy_c111, Lx, Ly, Lz, z_interface, None, f"above{medium_above}below{medium_below}numbaLebedev_interface_vyC111_t{t}N{N}.png")
            # plot_three_planes_2d(vz_c111, Lx, Ly, Lz, z_interface, None, f"above{medium_above}below{medium_below}numbaLebedev_interface_vzC111_t{t}N{N}.png")

            plot_three_planes_2d(vx_c111, Lx, Ly, Lz, z_interface, None, f'{medium_above}_{medium_below}interfaceLeb{source_type}_vx.png')#f"above{medium_above}below{medium_below}numbaLebedev_inteface_vxC111_t{t}N{N}.png")
            plot_three_planes_2d(vy_c111, Lx, Ly, Lz, z_interface, None,  f'{medium_above}_{medium_below}interfaceLeb{source_type}_vy.png')#f"above{medium_above}below{medium_below}numbaLebedev_interface_vyC111_t{t}N{N}.png")
            plot_three_planes_2d(vz_c111, Lx, Ly, Lz, z_interface, None,  f'{medium_above}_{medium_below}interfaceLeb{source_type}_vz.png')#f"above{medium_above}below{medium_below}numbaLebedev_interface_vzC111_t{t}N{N}.png")

            # plot_lisitsa_pretty_fullplanes(
            # vx_c111,
            # Lx, Ly, Lz,
            # x0=Lx/2, y0=Ly/2, z0=Lz/2,
            # title=None,
            # savepath=f"above{medium_above}below{medium_below}_{source_type}numbastagg3d_vx_fullplanes_t{t}_N{N}.png",
            # upsample=2, show_plane_borders=True
            # )
            # plot_lisitsa_pretty_fullplanes(
            # vy_c111,
            # Lx, Ly, Lz,
            # x0=Lx/2, y0=Ly/2, z0=Lz/2,
            # title=None,
            # savepath=f"above{medium_above}below{medium_below}_{source_type}numbastagg3d_vy_fullplaness_t{t}_N{N}.png",
            # upsample=2, show_plane_borders=True
            # )
            # plot_lisitsa_pretty_fullplanes(
            # vz_c111,
            # Lx, Ly, Lz,
            # x0=Lx/2, y0=Ly/2, z0=Lz/2,
            # title=None,
            # savepath=f"above{medium_above}below{medium_below}_{source_type}numbastagg3d_vz_fullplaness_t{t}_N{N}.png",
            # upsample=2, show_plane_borders=True
            # )

            ###### END TIMER ######
            end = time.perf_counter()

            print(f'Elapsed time (with compilation) = {end-start}')
            break

        #VELOCITY ORDERING (111), (100), (010), (001)
        vx_copies = []
        vy_copies = []
        vz_copies = []
        for ci in range(4):
            vcluster = VCLUSTERS[ci]   
            vx_copies.append(V[ci][VX])
            vy_copies.append(V[ci][VY])
            vz_copies.append(V[ci][VZ])
        
    return vx_copies, vy_copies, vz_copies

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    
    
    # z < z_interface
    VP_below = 3.0
    VS_below = 1.0
   
    # z > z_interface
    VP_above = 3.0
    VS_above = 1.0
    
    medium_below = "VTI"   # "ISO", "VTI", "TTI" or "ANI"
    medium_above = "VTI"   # "ISO", "VTI", "TTI" or "ANI"

    result = run_Lebedev_simulation(
        N = 201,
        L = 3.0,
        T = 1.0,
        rho = 2.5,
        VP_above=VP_above,
        VP_below=VP_below,
        VS_above=VS_above,
        VS_below=VS_below,
        medium_above = medium_above,
        medium_below=medium_below
    )
    print('Finished simulation')



if __name__ == "__main__":
    main()








