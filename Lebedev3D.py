import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

from scipy.sparse import kron, csc_matrix, eye, bmat, diags, block_diag
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import imageio.v2 as imageio
import pyvista as pv
import scipy


# ============================================================
# 3D isotropic elastic wave simulation on a single staggered grid
# with cluster/grid logic and block-wise operators
# ============================================================








#In VTI (Vertically Transversely Isotropic media) --> Possess qP-wave, qSV- or slow S-wave (complicated structure), and qSH- or fast S-wave (elliptical)
#Thus, expect S-wave splitting.
# ------------------------------------------------------------
# Binary logic for clusters / grids
# ------------------------------------------------------------

#tuple to represent the cluster/grid notation
Bit3 = Tuple[int, int, int]

#XOR function
def xor_bit(a: Bit3, b: Bit3) -> Bit3:
    return (a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2])

#shifts as defined in LebedevElastics
VELOCITY_SHIFT: Dict[str, Bit3] = { #velocity shift in axis direction
    "x": (1, 0, 0),
    "y": (0, 1, 0),
    "z": (0, 0, 1),
}

STRESS_SHIFT: Dict[str, Bit3] = { #shear stress shift in "missing orthogonal" direction, normal stresses (1, 1, 1)
    "xx": (1, 1, 1),
    "yy": (1, 1, 1),
    "zz": (1, 1, 1),
    "yz": (1, 0, 0),
    "xz": (0, 1, 0),
    "xy": (0, 0, 1),
}


#dataclass that tries to emulate C struct with (family, component, cluster)
#certain functions are defined as properties of items of this class, these include: 
# 1. computing the shift, 
# 2. computing the grid,
# 3. computing the dual cluster
@dataclass(frozen=True)
class FieldID:
    family: str        # "v" or "sigma"
    component: str     # "x", "z", "xx", "zz", "xz"
    cluster: Bit3


    #1. compute the shift
    @property
    def shift(self) -> Bit3:
        if self.family == "v":
            return VELOCITY_SHIFT[self.component]
        if self.family == "sigma":
            return STRESS_SHIFT[self.component]
        raise ValueError(f"Unknown family: {self.family}")

    #2. compute the grid by: g = XOR(cluster, shift)
    @property
    def grid(self) -> Bit3:
        return xor_bit(self.cluster, self.shift)

    #3. compute dual cluster
    @property
    def dual_cluster(self) -> Bit3:
        return xor_bit(self.cluster, (1, 1, 1))


#define this class in an effort to avoid typical index slicing, contains the offset, size and shape
@dataclass(frozen=True)
class Block:
    offset: int
    size: int
    shape: tuple[int, int, int]

#this class contains the fields, and their blocks
@dataclass(frozen=True)
class Layout3D:
    sigma_fields: dict
    v_fields: dict
    sigma_blocks: dict
    v_blocks: dict


# ------------------------------------------------------------
# Layout from derived grids
# ------------------------------------------------------------

#function that determines the shape of matrices based on the grid
def shape_from_grid(grid: Bit3, Nx: int, Ny: int, Nz: int):
    gx, gy, gz = grid
    nx = Nx if gx == 0 else Nx - 1
    ny = Ny if gy == 0 else Ny - 1
    nz = Nz if gz == 0 else Nz - 1
    return (nx, ny, nz), nx * ny * nz

#using stress-velocity cluster, define all variable fields and the stress/vel blocks
def build_layout_from_clusters(SCL: Bit3, VCL: Bit3, Nx: int, Ny: int, Nz: int) -> Layout3D:

    sigma_fields = { #define the 3 stress unknowns with their cluster
        "xx": FieldID("sigma", "xx", SCL),
        "yy": FieldID("sigma", "yy", SCL),
        "zz": FieldID("sigma", "zz", SCL),
        "yz": FieldID("sigma", "yz", SCL),
        "xz": FieldID("sigma", "xz", SCL),
        "xy": FieldID("sigma", "xy", SCL),
    }
    v_fields = { #define the 2 velocity unknowns with their cluster (dual to SCL)
        "x": FieldID("v", "x", VCL),
        "y": FieldID("v", "y", VCL),
        "z": FieldID("v", "z", VCL),
    }


    #fill the sigma/velocity blocks with data from the grids
    sigma_blocks = {}
    offset = 0
    for key in ['xx', 'zz', 'yy', 'yz', 'xz', 'xy']:
        shape, size = shape_from_grid(sigma_fields[key].grid, Nx, Ny, Nz)
        sigma_blocks[key] = Block(offset, size, shape)
        offset += size

    v_blocks = {}
    offset = 0
    for key in ['x', 'y', 'z']:
        shape, size = shape_from_grid(v_fields[key].grid, Nx, Ny, Nz)
        v_blocks[key] = Block(offset, size, shape)
        offset += size

    return Layout3D( #returns object that contains the fields and blocks for this cluster/grid layout
        sigma_fields=sigma_fields,
        v_fields=v_fields,
        sigma_blocks=sigma_blocks,
        v_blocks=v_blocks,
    )

#function that extracts and reshapes the variable from the block
def extract_field(vec: np.ndarray, block: Block) -> np.ndarray:
    sl = slice(block.offset, block.offset + block.size)
    return vec[sl].reshape(block.shape, order='F')




# def lebedev_permutation():
#     P = np.zeros((24,24))

#     def idx(cluster, comp):
#         base = {
#             (0,0,0): 0,
#             (0,1,1): 6,
#             (1,0,1): 12,
#             (1,1,0): 18
#         }[cluster]

#         comp_id = {"xx":0,"yy":1,"zz":2,"yz":3,"xz":4,"xy":5}[comp]
#         return base + comp_id

#     rows = [
#         # grid 111
#         ((0,0,0),"xx"), ((0,0,0),"yy"), ((0,0,0),"zz"),
#         ((0,1,1),"yz"), ((1,0,1),"xz"), ((1,1,0),"xy"),

#         # grid 100
#         ((0,1,1),"xx"), ((0,1,1),"yy"), ((0,1,1),"zz"),
#         ((0,0,0),"yz"), ((1,1,0),"xz"), ((1,0,1),"xy"),

#         # grid 010
#         ((1,0,1),"xx"), ((1,0,1),"yy"), ((1,0,1),"zz"),
#         ((1,1,0),"yz"), ((0,0,0),"xz"), ((0,1,1),"xy"),

#         # grid 001
#         ((1,1,0),"xx"), ((1,1,0),"yy"), ((1,1,0),"zz"),
#         ((1,0,1),"yz"), ((0,1,1),"xz"), ((0,0,0),"xy"),
#     ]

#     for i,(c,comp) in enumerate(rows):
#         P[i, idx(c,comp)] = 1

#     return P





def grid_cluster_permutation(layouts):
    #Grid ordering: [000], [011], [101], [110], differentiation circle logic
    #Cluster ordering: (111), (001), (010), (100)
    s_comps = ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']
    v_comps = ['x', 'y', 'z']
    stress_blocks = []
    vel_blocks = []
    for layout in layouts:
        for comp in s_comps:
            stress_blocks.append(layout.sigma_blocks[comp])
        for comp in v_comps:
            vel_blocks.append(layout.v_blocks[comp])
    
    idx_s = [0,1,2,6,12,18,9,10,11,5,23,17,14,15,16,22,4,8,19,20,21,13,7,3]
    idx_v = [3,7,11,2,10,6,9,1,5,8,4,0]
 
    n = len(stress_blocks)
    m = len(vel_blocks)
    Ps_blocks = [[None for _ in range(n)] for _ in range(n)] #create empty blocks for P
    Pv_blocks = [[None for _ in range(m)] for _ in range(m)] #create empty blocks for P
    for i in range(n):
        j = idx_s[i]
        bj = stress_blocks[j]
        Ps_blocks[i][j] = eye(bj.size, format='csc')
    P_sigma = bmat(Ps_blocks, format='csc')

    for i in range(m):
        j = idx_v[i]
        bj = vel_blocks[j]
        Pv_blocks[i][j] = eye(bj.size, format='csc')
    P_v = bmat(Pv_blocks, format='csc')

    return P_sigma, P_v
    




def velocity_permutation_mat():
    #Grid ordering: [000], [011], [101], [110], differentiation circle logic
    #Cluster ordering: (111), (001), (010), (100)
    P = np.zeros((12,12))
    idxs = [3,7,11,2,10,6,9,1,5,8,4,0]
    for row in range(len(P)):
        P[row, idxs[row]] = 1
    return P


    # P = np.array([
    #     #grid [111]
    #     [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],

    #     #grid [100]
    #     [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],

    #     #grid [010]
    #     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    #     [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],

    #     #grid [001]
    #     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        
    # ])






















#function that creates Bidiagonal matrix that computes differences between neighbours
def bidiag(N: int):
    interior = [-np.ones(N - 1), np.ones(N - 1)]
    offsets = [0, 1]
    return diags(interior, offsets, shape=(N - 1, N), format="csc")



#function that creates 1D primary and dual derivative operator 
def diff_ops_1D(N: int, h_p, h_d):
    if np.isscalar(h_p):
        h_p = h_p * np.ones(N - 1)
    else:
        h_p = np.asarray(h_p, dtype=float)
    if np.isscalar(h_d):
        h_d = h_d * np.ones(N)
    else:
        h_d = np.asarray(h_d, dtype=float)
    
    B = bidiag(N)
    W_p_inv = diags(1.0 / h_p, 0, format="csc")
    W_d_inv = diags(1.0 / h_d, 0, format="csc")

    D_p = W_p_inv @ B
    D_d = -W_d_inv @ B.T

    return D_p, D_d


#function that expands derivative ops to 2D with appropiate sizes from grids/clusters
def derivative_3d(field: FieldID, axis: str, Nx: int, Ny: int, Nz: int, hx_p, hx_d, hy_p, hy_d, hz_p, hz_d):
    X_p, X_d = diff_ops_1D(Nx, hx_p, hx_d)
    Y_p, Y_d = diff_ops_1D(Ny, hy_p, hy_d)
    Z_p, Z_d = diff_ops_1D(Nz, hz_p, hz_d)

    #define I matrices for primary and dual (in interior)
    Ix_p = eye(Nx, format="csc")
    Iy_p = eye(Ny, format="csc")
    Iz_p = eye(Nz, format="csc")
    Ix_d = eye(Nx - 1, format="csc")
    Iy_d = eye(Ny - 1, format="csc")
    Iz_d = eye(Nz - 1, format="csc")

    gx, gy, gz = field.grid 

    #for x,y,z derivative, depending on the grid, choose diff op and correct sizes for I
    if axis == "x":
        Dx = X_p if gx == 0 else X_d
        Iy = Iy_p if gy == 0 else Iy_d
        Iz = Iz_p if gz == 0 else Iz_d
        return kron(Iz, kron(Iy, Dx, format="csc"), format='csc')

    if axis == "y":
        Dy = Y_p if gy == 0 else Y_d
        Ix = Ix_p if gx == 0 else Ix_d
        Iz = Iz_p if gz == 0 else Iz_d
        return kron(Iz, kron(Dy, Ix, format="csc"), format='csc')
       

    if axis == "z":
        Dz = Z_p if gz == 0 else Z_d
        Iy = Iy_p if gy == 0 else Iy_d
        Ix = Ix_p if gx == 0 else Ix_d
        return kron(Dz, kron(Iy, Ix, format="csc"), format='csc')
       

    raise ValueError("axis must be 'x', 'z' or 'y'")


#function that assembles diff ops into block form
def block_diff_ops(SCL: Bit3, VCL: Bit3, Nx: int, Ny: int, Nz: int, hx_p, hx_d, hy_p, hy_d, hz_p, hz_d):
    
    layout = build_layout_from_clusters(SCL, VCL, Nx, Ny, Nz) #returns Layout2D object

    #extract fields/blocks from Layout2D object
    sigma_fields = layout.sigma_fields
    v_fields = layout.v_fields
    #sigma_blocks = layout.sigma_blocks
    #v_blocks = layout.v_blocks

    #elements of D_sigma, norm
    Dx_sxx = derivative_3d(sigma_fields["xx"], "x", Nx, Ny, Nz, hx_p, hx_d, hy_p, hy_d, hz_p, hz_d)
    Dy_syy = derivative_3d(sigma_fields["yy"], "y", Nx, Ny, Nz, hx_p, hx_d, hy_p, hy_d, hz_p, hz_d)
    Dz_szz = derivative_3d(sigma_fields["zz"], "z", Nx, Ny, Nz, hx_p, hx_d, hy_p, hy_d, hz_p, hz_d)
    
    #elements of D_sigma, shear
    Dz_sxz = derivative_3d(sigma_fields['xz'], 'z', Nx, Ny, Nz, hx_p, hx_d, hy_p, hy_d, hz_p, hz_d)
    Dy_sxy = derivative_3d(sigma_fields['xy'], 'y', Nx, Ny, Nz, hx_p, hx_d, hy_p, hy_d, hz_p, hz_d)
    Dz_syz = derivative_3d(sigma_fields['yz'], 'z', Nx, Ny, Nz, hx_p, hx_d, hy_p, hy_d, hz_p, hz_d)
    Dx_sxy = derivative_3d(sigma_fields['xy'], 'x', Nx, Ny, Nz, hx_p, hx_d, hy_p, hy_d, hz_p, hz_d)
    Dy_syz = derivative_3d(sigma_fields['yz'], 'y', Nx, Ny, Nz, hx_p, hx_d, hy_p, hy_d, hz_p, hz_d)
    Dx_sxz = derivative_3d(sigma_fields['xz'], 'x', Nx, Ny, Nz, hx_p, hx_d, hy_p, hy_d, hz_p, hz_d)

    #zero entries
    # Z_vx_sigma_zz = csc_matrix((v_blocks["x"].size, sigma_blocks["zz"].size))
    # Z_vz_sigma_xx = csc_matrix((v_blocks["z"].size, sigma_blocks["xx"].size))

    #construct D_sigma
    D_sigma = bmat([
        [Dx_sxx, None, None, None, Dz_sxz, Dy_sxy],
        [None, Dy_syy, None, Dz_syz, None, Dx_sxy],
        [None, None, Dz_szz, Dy_syz, Dx_sxz, None],
    ], format="csc")

    #elements of D_v
    Dx_vx = derivative_3d(v_fields['x'], 'x', Nx, Ny, Nz, hx_p, hx_d, hy_p, hy_d, hz_p, hz_d)
    Dy_vy = derivative_3d(v_fields['y'], 'y', Nx, Ny, Nz, hx_p, hx_d, hy_p, hy_d, hz_p, hz_d)
    Dz_vz = derivative_3d(v_fields['z'], 'z', Nx, Ny, Nz, hx_p, hx_d, hy_p, hy_d, hz_p, hz_d)

    Dz_vy = derivative_3d(v_fields['y'], 'z', Nx, Ny, Nz, hx_p, hx_d, hy_p, hy_d, hz_p, hz_d)
    Dy_vz = derivative_3d(v_fields['z'], 'y', Nx, Ny, Nz, hx_p, hx_d, hy_p, hy_d, hz_p, hz_d)
    Dz_vx = derivative_3d(v_fields['x'], 'z', Nx, Ny, Nz, hx_p, hx_d, hy_p, hy_d, hz_p, hz_d)
    Dx_vz = derivative_3d(v_fields['z'], 'x', Nx, Ny, Nz, hx_p, hx_d, hy_p, hy_d, hz_p, hz_d)
    Dy_vx = derivative_3d(v_fields['x'], 'y', Nx, Ny, Nz, hx_p, hx_d, hy_p, hy_d, hz_p, hz_d)
    Dx_vy = derivative_3d(v_fields['y'], 'x', Nx, Ny, Nz, hx_p, hx_d, hy_p, hy_d, hz_p, hz_d)


    #construct D_v
    D_v = bmat([
        [Dx_vx, None, None],
        [None, Dy_vy, None],
        [None, None, Dz_vz],
        [None, Dz_vy, Dy_vz],
        [Dz_vx, None, Dx_vz],
        [Dy_vx, Dx_vy, None],
     
    ], format="csc")

    return D_sigma, D_v, layout


#function that defines density matrix and stiffness matrix with Lame param
def material_matrix(layout: Layout3D, rho: float, lam: float, mu: float, C_mat: np.ndarray):
    #isotropic homogeneous medium
    N_vx = layout.v_blocks["x"].size
    N_vy = layout.v_blocks['y'].size
    N_vz = layout.v_blocks["z"].size

    N_sxx = layout.sigma_blocks["xx"].size
    N_syy = layout.sigma_blocks["yy"].size
    N_szz = layout.sigma_blocks["zz"].size

    N_syz = layout.sigma_blocks["yz"].size
    N_sxz = layout.sigma_blocks["xz"].size
    N_sxy = layout.sigma_blocks["xy"].size

    M_rho_vx = rho * eye(N_vx, format="csc")
    M_rho_vy = rho * eye(N_vy, format="csc")
    M_rho_vz = rho * eye(N_vz, format="csc")
    M_rho = block_diag((M_rho_vx, M_rho_vy, M_rho_vz), format="csc")


    #extract values of stiffness matrix
    C11, C12, C13, C14, C15, C16 = C_mat[0,0], C_mat[0,1], C_mat[0,2], C_mat[0,3], C_mat[0,4], C_mat[0,5]
    C22, C23, C24, C25, C26 = C_mat[1,1], C_mat[1,2], C_mat[2,3], C_mat[2,4], C_mat[2,5]
    C33, C34, C35, C36 = C_mat[2,2], C_mat[2,3], C_mat[2,4], C_mat[2,5]
    C44, C45, C46 = C_mat[3,3], C_mat[3,4], C_mat[3,5]
    C55, C56 = C_mat[4,4], C_mat[4,5]
    C66 = C_mat[5,5]


    # VTI / isotropic case
    C11_blk = C11 * eye(N_sxx, format='csc')
    C12_blk = C12 * eye(N_syy, format='csc')

    C44_blk = C44 * eye(N_syz, format='csc')
    C55_blk = C55 * eye(N_sxz, format='csc')
    C66_blk = C66 * eye(N_sxy, format='csc')

    # C13_blk = C13 * eye(N_szz, format='csc')
    # C33_blk = C33 * eye(N_szz, format='csc')
    # C55_blk = C55 * eye(N_sxz, format='csc')
  

  
    C = bmat([
        [C11_blk, C12_blk, C12_blk, None, None, None],
        [C12_blk, C11_blk, C12_blk, None, None, None],
        [C12_blk, C12_blk, C11_blk, None, None, None],
        [None, None, None, C44_blk, None, None],
        [None, None, None, None, C55_blk, None],
        [None, None, None, None, None, C66_blk],
    ], format="csc")

    return M_rho, C


# ------------------------------------------------------------
# Initial conditions / sources
# ------------------------------------------------------------

def initial_condition_sigma(sigma, layout: Layout3D, Lx, Ly, Lz, Nx, Ny, Nz, A, s):
    xx_block = layout.sigma_blocks["xx"]
    yy_block = layout.sigma_blocks["yy"]
    zz_block = layout.sigma_blocks["zz"]

    # xx_shape = xx_block.shape
    # yy_shape = yy_block.shape
    # zz_shape = zz_block.shape

    # x_xx = np.linspace(0, Lx, xx_shape[0])
    # y_xx = np.linspace(0, Ly, xx_shape[1])
    # z_xx = np.linspace(0, Lz, xx_shape[2])
    # X_xx, Y_xx, Z_xx = np.meshgrid(x_xx, y_xx, z_xx, indexing="ij")

    # x_zz = np.linspace(0, Lx, zz_shape[0])
    # z_zz = np.linspace(0, Lz, zz_shape[1])
    # X_zz, Z_zz = np.meshgrid(x_zz, z_zz, indexing="ij")

    x0 = Lx / 2
    y0 = Ly / 2
    z0 = Lz / 2
   
    x = np.linspace(0, Lx, xx_block.shape[0])
    y = np.linspace(0, Ly, xx_block.shape[1])
    z = np.linspace(0, Lz, xx_block.shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    g = np.exp(-((X - x0)**2 + (Y - y0)**2 + (Z - z0)**2) / (2 * s**2))
   

    # g_xx = A * np.exp(-((X_xx - x0) ** 2 + (Z_xx - z0) ** 2) / (2 * s ** 2))
    # g_zz = A * np.exp(-((X_zz - x0) ** 2 + (Z_zz - z0) ** 2) / (2 * s ** 2))

    sigma[xx_block.offset:xx_block.offset + xx_block.size] = g.flatten(order="F")
    sigma[yy_block.offset:yy_block.offset + yy_block.size] = g.flatten(order="F")
    sigma[zz_block.offset:zz_block.offset + zz_block.size] = -g.flatten(order="F")
    return sigma


def gaussian_on_block_grid(block: Block, Lx, Ly, Lz, x0, y0, z0, s):
    x = np.linspace(0, Lx, block.shape[0])
    y = np.linspace(0, Ly, block.shape[1])
    z = np.linspace(0, Lz, block.shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    g = np.exp(-((X - x0) ** 2 + (Y-y0)**2 +(Z - z0) ** 2) / (2 * s ** 2))
    return g.flatten(order="F")


def source(layout: Layout3D, Lx, Ly, Lz, x0, y0, z0, s, amp):
    N_sigma = sum(block.size for block in layout.sigma_blocks.values())
    q_sigma = np.zeros(N_sigma)

    g_xz = gaussian_on_block_grid(layout.sigma_blocks["xz"], Lx, Ly, Lz, x0, y0, z0, s)
    g_xx = gaussian_on_block_grid(layout.sigma_blocks["xx"], Lx, Ly, Lz, x0, y0, z0, s)
    g_yy = gaussian_on_block_grid(layout.sigma_blocks["yy"], Lx, Ly, Lz, x0, y0, z0, s)
    g_zz = gaussian_on_block_grid(layout.sigma_blocks["zz"], Lx, Ly, Lz, x0, y0, z0, s)

    xz_block = layout.sigma_blocks["xz"]
    xx_block = layout.sigma_blocks["xx"]
    yy_block = layout.sigma_blocks["yy"]
    zz_block = layout.sigma_blocks["zz"]
  
    q_sigma[xz_block.offset:xz_block.offset + xz_block.size] = amp * g_xz

    # q_sigma[xx_block.offset:xx_block.offset + xx_block.size] = amp * g_xx
    # q_sigma[yy_block.offset:yy_block.offset + yy_block.size] = amp * g_yy
    # q_sigma[zz_block.offset:zz_block.offset + zz_block.size] = amp * g_zz

    return q_sigma


def ricker(t, f0, t0):
    a = np.pi * f0 * (t - t0)
    return (1.0 - 2.0 * a ** 2) * np.exp(-a ** 2)


# ------------------------------------------------------------
# Plotting / GIF
# ------------------------------------------------------------



from matplotlib import cm, colors

def _slice_at_position(field3, plane, pos, Lx, Ly, Lz):
    """
    field3 shape: (nx, ny, nz)
    pos: physical coordinate of the slice
    returns:
        slab   : 2D array
        coord  : actual physical coordinate used
    """
    nx, ny, nz = field3.shape

    if plane == "xy":
        z = np.linspace(0.0, Lz, nz)
        k = int(np.argmin(np.abs(z - pos)))
        return field3[:, :, k], z[k]

    elif plane == "xz":
        y = np.linspace(0.0, Ly, ny)
        j = int(np.argmin(np.abs(y - pos)))
        return field3[:, j, :], y[j]

    elif plane == "yz":
        x = np.linspace(0.0, Lx, nx)
        i = int(np.argmin(np.abs(x - pos)))
        return field3[i, :, :], x[i]

    else:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")


def plot_intersecting_planes(
    field3,
    Lx, Ly, Lz,
    plane1=("xy", None),
    plane2=("xz", None),
    source=None,
    cmap="seismic",
    title=None,
    savepath=None,
    show_box=True,
    elev=24,
    azim=-58,
):
    """
    Plot two intersecting slice planes of a 3D scalar field.

    Example:
        plot_intersecting_planes(
            vx, Lx, Ly, Lz,
            plane1=("xy", Lz/2),
            plane2=("xz", Ly/2),
            source=(Lx/2, Ly/2, Lz/2),
            title="vx on intersecting planes"
        )
    """
    nx, ny, nz = field3.shape

    # default slice positions: center planes
    defaults = {
        "xy": Lz / 2.0,
        "xz": Ly / 2.0,
        "yz": Lx / 2.0,
    }

    p1, pos1 = plane1
    p2, pos2 = plane2
    if pos1 is None:
        pos1 = defaults[p1]
    if pos2 is None:
        pos2 = defaults[p2]

    slab1, pos1_used = _slice_at_position(field3, p1, pos1, Lx, Ly, Lz)
    slab2, pos2_used = _slice_at_position(field3, p2, pos2, Lx, Ly, Lz)

    vmax = max(np.max(np.abs(slab1)), np.max(np.abs(slab2)))
    if vmax == 0:
        vmax = 1.0
    norm = colors.Normalize(vmin=-vmax, vmax=vmax)
    cmap_obj = cm.get_cmap(cmap)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # ---------- first plane ----------
    if p1 == "xy":
        x = np.linspace(0.0, Lx, slab1.shape[0])
        y = np.linspace(0.0, Ly, slab1.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")
        Z = np.full_like(X, pos1_used)
        FC = cmap_obj(norm(slab1))
        ax.plot_surface(X, Y, Z, facecolors=FC, shade=False, linewidth=0)

    elif p1 == "xz":
        x = np.linspace(0.0, Lx, slab1.shape[0])
        z = np.linspace(0.0, Lz, slab1.shape[1])
        X, Z = np.meshgrid(x, z, indexing="ij")
        Y = np.full_like(X, pos1_used)
        FC = cmap_obj(norm(slab1))
        ax.plot_surface(X, Y, Z, facecolors=FC, shade=False, linewidth=0)

    elif p1 == "yz":
        y = np.linspace(0.0, Ly, slab1.shape[0])
        z = np.linspace(0.0, Lz, slab1.shape[1])
        Y, Z = np.meshgrid(y, z, indexing="ij")
        X = np.full_like(Y, pos1_used)
        FC = cmap_obj(norm(slab1))
        ax.plot_surface(X, Y, Z, facecolors=FC, shade=False, linewidth=0)

    # ---------- second plane ----------
    if p2 == "xy":
        x = np.linspace(0.0, Lx, slab2.shape[0])
        y = np.linspace(0.0, Ly, slab2.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")
        Z = np.full_like(X, pos2_used)
        FC = cmap_obj(norm(slab2))
        ax.plot_surface(X, Y, Z, facecolors=FC, shade=False, linewidth=0)

    elif p2 == "xz":
        x = np.linspace(0.0, Lx, slab2.shape[0])
        z = np.linspace(0.0, Lz, slab2.shape[1])
        X, Z = np.meshgrid(x, z, indexing="ij")
        Y = np.full_like(X, pos2_used)
        FC = cmap_obj(norm(slab2))
        ax.plot_surface(X, Y, Z, facecolors=FC, shade=False, linewidth=0)

    elif p2 == "yz":
        y = np.linspace(0.0, Ly, slab2.shape[0])
        z = np.linspace(0.0, Lz, slab2.shape[1])
        Y, Z = np.meshgrid(y, z, indexing="ij")
        X = np.full_like(Y, pos2_used)
        FC = cmap_obj(norm(slab2))
        ax.plot_surface(X, Y, Z, facecolors=FC, shade=False, linewidth=0)

    # source / intersection marker
    if source is not None:
        ax.scatter(source[0], source[1], source[2], color="k", s=40)

    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(0, Lz)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if not show_box:
        ax.set_axis_off()

    ax.view_init(elev=elev, azim=azim)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.75, pad=0.08)

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")

    plt.show()




def PPW(trace, dt, h, V_P, V_S, cutoff):
    F = np.fft.rfft(trace)
    freq = np.fft.rfftfreq(len(trace), d=dt)
    A = np.abs(F)
    plt.plot(freq,A)
    plt.show()

    #dominant freq
    f_dom = freq[np.argmax(A)]
    Amax = np.max(A)
    mask =  A > cutoff * Amax #larger than 10% of Max 
    if np.any(mask):
        f_max = freq[mask][-1] #freq where amplitude is 10% of max
    else:
        f_max = f_dom 
    # (center freq) wavelengths
    lambda_P = V_P / f_dom #period times velocity = wavelength
    lambda_S = V_S / f_dom

    lambda_min_P = V_P / f_max
    lambda_min_S = V_S / f_max

    # PPW
    PPW_P = lambda_P / h
    PPW_S = lambda_S / h

    PPW_P_min = lambda_min_P / h
    PPW_S_min = lambda_min_S / h

    print("Dominant frequency:", f_dom)
    print("Max frequency (band limit):", f_max)

    print(f"P-wave λ = {lambda_P}, PPW = {PPW_P}")
    print(f"S-wave λ = {lambda_S}, PPW = {PPW_S}")

    print(f"P-wave min λ = {lambda_min_P}, PPW(min) = {PPW_P_min}")
    print(f"S-wave min λ = {lambda_min_S}, PPW(min) = {PPW_S_min}")

    return PPW_P_min, PPW_S_min








def lebedev_derivative_operators(SCL_lst, VCL_lst, Nx, Ny, Nz, hx_p, hx_d, hy_p, hy_d, hz_p, hz_d):
    #Build the block diagonal Lebedev derivative operators D_sigma, D_v
    D_s_list = []
    D_v_list = []
    layouts = []
    for scl, vcl in zip(SCL_lst, VCL_lst):
        Ds, Dv, layout = block_diff_ops(scl, vcl, Nx, Ny, Nz, hx_p, hx_d, hy_p, hy_d, hz_p, hz_d)
        D_s_list.append(Ds)
        D_v_list.append(Dv)
        layouts.append(layout)
    D_sigma = block_diag(D_s_list, format='csc')
    D_v = block_diag(D_v_list, format='csc')
    return D_sigma, D_v, layouts



def diag_field(field):
    #convert a 3d field into a diagonal sprace matrix using F ordering
    return diags(field.reshape(-1,order='F'), 0, format='csc')


def isotropic_stiffness_block(lam_arr, mu_arr):
    L = diag_field(lam_arr)
    M = diag_field(mu_arr)
    LP2M = L + 2*M
    return bmat([
        [LP2M, L, L, None, None, None],
        [L, LP2M, L, None, None, None],
        [L, L , LP2M, None, None, None],
        [None, None, None, M, None, None],
        [None, None, None, None, M, None],
        [None, None, None, None, None, M]
    ], format='csc')

def average_axis(a, axis, mode): 
    #averages neighboring values of arr along one axis, mode - arithmetic/harmonic
    s0 = [slice(None)] * a.ndim #select all indices
    s1 = [slice(None)] * a.ndim #select all indices
    s0[axis] = slice(0,-1) #from start to end
    s1[axis] = slice(1,None) #from start+1 b
    a0 = a[tuple(s0)]
    a1 = a[tuple(s1)]
    if mode == 'A':
        return 0.5 * (a0 + a1)
    elif mode == 'H':
        return 2.0 / (1.0 / a0 + 1.0 / a1)
    else:
        raise ValueError('unknown mode')

def average_to_grid(field, source_grid, dest_grid, mode):
    out = field #field = rho / stiff
    for axis, (source, dest) in enumerate(zip(source_grid, dest_grid)):
        if source != dest:
            out = average_axis(out, axis, mode)
    return out






def iso_heterogeneous_medium_lebedev_material_matrix(layouts, Nx, Ny, Nz, rho, lam, mu):
    #define "raw" density and stiffness on [000], average to other grids
    #has sizes (Nx,Ny,Nz)
    rho0 = rho * np.ones((Nx, Ny, Nz))
    lam0 = lam * np.ones((Nx, Ny, Nz))
    mu0 = mu * np.ones((Nx, Ny, Nz))

    #heterogeneous test
    rho0[:, :, Nz//2:] *= 2
    #mu0[:, :, Nz//2:] *= 0.8


    #ARITHMETIC AVG OF RHO ONTO GRIDS 
    rho_000 = rho0
    rho_011 = average_to_grid(rho0, (0,0,0), (0,1,1), 'A')
    rho_101 = average_to_grid(rho0, (0,0,0), (1,0,1), 'A')
    rho_110 = average_to_grid(rho0, (0,0,0), (1,1,0), 'A')

    #HARMONIC AVG OF C ONTO GRIDS
    
    #isotropic case w/ Lamé parameters
    mu_111 = average_to_grid(mu0, (0,0,0), (1,1,1), 'H')
    mu_100 = average_to_grid(mu0, (0,0,0), (1,0,0), 'H')
    mu_010 = average_to_grid(mu0, (0,0,0), (0,1,0), 'H')
    mu_001 = average_to_grid(mu0, (0,0,0), (0,0,1), 'H')

    Mrho_inv_hat = block_diag([
        block_diag((
            diag_field(1.0 / rho_000),
            diag_field(1.0 / rho_000),
            diag_field(1.0 / rho_000),
        ), format="csc"),

        block_diag((
            diag_field(1.0 / rho_011),
            diag_field(1.0 / rho_011),
            diag_field(1.0 / rho_011),
        ), format="csc"),

        block_diag((
            diag_field(1.0 / rho_101),
            diag_field(1.0 / rho_101),
            diag_field(1.0 / rho_101),
        ), format="csc"),

        block_diag((
            diag_field(1.0 / rho_110),
            diag_field(1.0 / rho_110),
            diag_field(1.0 / rho_110),
        ), format="csc"),
    ], format="csc")


    lam_111 = average_to_grid(lam0, (0,0,0), (1,1,1), 'H')
    lam_100 = average_to_grid(lam0, (0,0,0), (1,0,0), 'H')
    lam_010 = average_to_grid(lam0, (0,0,0), (0,1,0), 'H')
    lam_001 = average_to_grid(lam0, (0,0,0), (0,0,1), 'H')

    

    C_hat = block_diag([
        isotropic_stiffness_block(lam_111, mu_111),
        isotropic_stiffness_block(lam_100, mu_100),
        isotropic_stiffness_block(lam_010, mu_010),
        isotropic_stiffness_block(lam_001, mu_001),
    ], format="csc")

    return C_hat, Mrho_inv_hat



def const_medium_lebedev_material_matrix(layouts, Nx, Ny, Nz, rho, C_mat):

    layout0 = layouts[0] #cluster (000)

    #get stress grid sizes from (000) (from differentiation circle)
    N111 = layout0.sigma_blocks['xx'].size
    N100 = layout0.sigma_blocks['yz'].size
    N010 = layout0.sigma_blocks['xz'].size
    N001 = layout0.sigma_blocks['xy'].size

    C_loc = csc_matrix(C_mat)

    #expand stiffness matrix
    C_hat = block_diag([
        kron(C_loc, eye(N111, format='csc')),
        kron(C_loc, eye(N100, format='csc')),
        kron(C_loc, eye(N010, format='csc')),
        kron(C_loc, eye(N001, format='csc')),
    ], format='csc')

    #get velocity grid sizes from grids
    _, N000 = shape_from_grid((0,0,0), Nx, Ny, Nz)
    _, N011 = shape_from_grid((0,1,1), Nx, Ny, Nz)
    _, N101 = shape_from_grid((1,0,1), Nx, Ny, Nz)
    _, N110 = shape_from_grid((1,1,0), Nx, Ny, Nz)

    Mrho_inv_hat = block_diag([
        (1.0 / rho) * eye(3*N000, format='csc'),
        (1.0 / rho) * eye(3*N011, format='csc'),
        (1.0 / rho) * eye(3*N101, format='csc'),
        (1.0 / rho) * eye(3*N110, format='csc'),
    ], format='csc')

    return C_hat, Mrho_inv_hat





def simple_norm(v, sigma):
    return np.sqrt(np.dot(v, v) + np.dot(sigma, sigma))


def weighted_energy_general(v, sigma, Wv, Mrho, Ws, MCinv):
    Ev = v @ (Wv @ (Mrho @ v))
    Es = sigma @ (Ws @ (MCinv @ sigma))
    return Ev + Es


def energy(v, sigma, rho, Cinv, h):
    Ev = h**3 * rho * np.dot(v, v)
    Es = h**3 * sigma @ (Cinv @ sigma)
    return Ev + Es


#function that does timestepping
def leapfrog(v, sigma, Lx, Ly, Lz, dt, Nt, Nx, Ny, Nz, rho, C_mat, V_P, V_S, n_plot, field_gif, SCL_lst, VCL_lst):
    #step sizes
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dz = Lz / (Nz - 1)

    hx_p, hx_d = dx, dx
    hy_p, hy_d = dy, dy
    hz_p, hz_d = dz, dz


    #build Lebedev D_sigma, D_v and cluster pair layouts
    D_sigma, D_v, layouts = lebedev_derivative_operators(SCL_lst, VCL_lst, Nx, Ny, Nz, hx_p, hx_d, hy_p, hy_d, hz_p, hz_d)
    
    #build permutation matrices
    P_sigma, P_v = grid_cluster_permutation(layouts)

    #build material operators
    C_hat, Mrho_inv_hat = const_medium_lebedev_material_matrix(layouts, Nx, Ny, Nz, rho, C_mat)
    #C_hat, Mrho_inv_hat = iso_heterogeneous_medium_lebedev_material_matrix(layouts, Nx, Ny, Nz, rho, lam, mu)

    N_v = D_v.shape[1]
    N_sigma = D_sigma.shape[1]

    if v is None or len(v) != N_v:
        v = np.zeros(N_v)
    if sigma is None or len(sigma) != N_sigma:
        sigma = np.zeros(N_sigma)




    # #random IC for stability testing
    # rng = np.random.default_rng(0)
    # v = 1e-6 * rng.standard_normal(N_v)
    # sigma = 1e-6 * rng.standard_normal(N_sigma)
    # Wv = dx**3 * eye(N_v)
    # Ws = dx**3 * eye(N_sigma)

    # D = D_sigma.tocsr()
    # abs_row_sums = np.abs(D).sum(axis=1).A.ravel()
    # abs_col_sums = np.abs(D).sum(axis=0).A.ravel()
    # col_nnz = np.diff(D.tocsc().indptr)
    # row_nnz = np.diff(D.indptr)

    # print("max row abs sum:", abs_row_sums.max())
    # print("max col abs sum:", abs_col_sums.max())
    # print("max nnz per row:", row_nnz.max())
    # print("max nnz per col:", col_nnz.max())
    

    # D = D_sigma.tocsr()
    # h = hx_p  # assuming uniform grid

    # # crude induced-norm bound
    # row_abs = np.abs(D).sum(axis=1).A.ravel()
    # col_abs = np.abs(D).sum(axis=0).A.ravel()
    # D_bound_induced = np.sqrt(row_abs.max() * col_abs.max())

    # print(f"||D_sigma||_2 <= {D_bound_induced:.12g}  (from sqrt(||D||_1 ||D||_inf))")
    # print(f"scaled constant C_induced = h * ||D_sigma||_bound = {h * D_bound_induced:.12g}")

    # # Gershgorin on G = D^T D
    # G = (D.T @ D).tocsr()
    # gersh_row_sums = np.abs(G).sum(axis=1).A.ravel()
    # gersh_max = gersh_row_sums.max()

    # D_bound_gersh = np.sqrt(gersh_max)
    # C_gersh = h * D_bound_gersh

    # print(f"Gershgorin bound for ||D_sigma||_2: {D_bound_gersh:.12g}")
    # print(f"scaled constant C_gersh = h * ||D_sigma||_bound = {C_gersh:.12g}")

    # # structural bounds
    # row_abs = np.abs(D).sum(axis=1).A.ravel()
    # col_abs = np.abs(D).sum(axis=0).A.ravel()

    # print("max row abs sum:", row_abs.max())
    # print("max col abs sum:", col_abs.max())

    # D_bound_induced = np.sqrt(row_abs.max() * col_abs.max())
    # print(f"||D_sigma||_2 <= {D_bound_induced:.12g}  (from sqrt(||D||_1 ||D||_inf))")
    # print(f"scaled constant C_induced = h * ||D_sigma||_bound = {h * D_bound_induced:.12g}")

    # # Gershgorin on D^T D
    # G = (D.T @ D).tocsr()
    # gersh_row_sums = np.abs(G).sum(axis=1).A.ravel()
    # gersh_max = gersh_row_sums.max()
    # D_bound_gersh = np.sqrt(gersh_max)

    # print(f"Gershgorin bound for ||D_sigma||_2: {D_bound_gersh:.12g}")
    # print(f"scaled constant C_gersh = h * ||D_sigma||_bound = {h * D_bound_gersh:.12g}")

    # # actual ||D_sigma||_2 from largest eigenvalue of D^T D
    # lam_max = scipy.sparse.linalg.eigsh(G, k=1, which="LM", return_eigenvectors=False)[0]
    # norm_D = np.sqrt(lam_max)

   
    # print(f"Actual ||D_sigma||_2: {norm_D:.12g}")
    # print(f"scaled constant C_exact = h * ||D_sigma||_2 = {h * norm_D:.12g}")

    # # homogeneous CFL estimate
    # max_eigval_C = np.max(np.linalg.eigvalsh(C_mat))
    # V_max = np.sqrt(max_eigval_C / rho)

    # dt_est = 2.0 / (V_max * norm_D)
    # print(f"Numerical homogeneous CFL estimate: dt <= {dt_est}")


    # c = 1.3
    # dt = c * dt_est
    # print(f'dt = {dt}')

    # T = 1 #final time
    # Nt = int(np.ceil(T / dt)) #number of time steps



    # # exact norm for small grids only
    # if D.shape[0] <= 4000 and D.shape[1] <= 4000:
    #     D_exact = np.linalg.svd(D.toarray(), compute_uv=False)[0]
    #     C_exact = h * D_exact
    #     print(f"Actual ||D_sigma||_2: {D_exact:.12g}")
    #     print(f"scaled constant C_exact = h * ||D_sigma||_2 = {C_exact:.12g}")


    q_v = np.zeros(N_v) #zero source term for velocity
    #q_sigma = np.zeros(N_sigma)

    x0, y0, z0 = Lx / 2, Ly / 2, Lz / 2
    s = 0.05

    layout_plot = layouts[0]
    v0_block = layout_plot.v_blocks['x']
    #PPW
    trace = np.zeros(Nt)
   
    writer = imageio.get_writer(f"{field_gif}_waveP{V_P}S{V_S}.gif", mode="I", fps=20)

    f0 = 3
    t0 = 1.5 / f0
    norm_hist = []


    for n in range(Nt):
        t = n * dt

        #source
        pulse = ricker(t, f0, t0)
        q_sigma = np.concatenate([source(layout, Lx, Ly, Lz, x0, y0, z0, s, pulse) for layout in layouts])

        #1. half-integer stress update 
        sigma = sigma + dt * (P_sigma.T @ (C_hat @ (P_sigma @ (q_sigma + D_v @ v)))) #minus sign factored out

        #2. integer velocity update
        v = v + dt * (P_v.T @ (Mrho_inv_hat @ (P_v @ (q_v + D_sigma @ sigma))))

        #if n % n_plot == 0:
        #norm_hist.append(weighted_energy_general(v, sigma, Wv, Mrho_inv_hat, Ws, C_hat))
        #norm_hist.append(energy(v, sigma, rho, C_hat, dx))
        


        if n == Nt-1:
           
            Nv0 = sum(b.size for b in layouts[0].v_blocks.values()) 
            v0 = v[:Nv0] # velocity in cluster 111
                        
           
            vx = extract_field(v0, layouts[0].v_blocks['x']) #updated field, vx(111) on [000]
            vz = extract_field(v0, layouts[0].v_blocks['z']) #updated field, vz(111) on [110]

            x0, y0, z0 = Lx / 2, Ly / 2, Lz / 2

            # plot_intersecting_planes(
            #     vz,
            #     Lx, Ly, Lz,
            #     plane1=("xy", z0),
            #     plane2=("xz", y0),
            #     source=(x0, y0, z0),
            #     title="vz on intersecting planes",
            #     savepath="vz_two_planes.png"
            #     )

            #plot_plane(vx, Lx, Ly, Lz, 'xy', None, 'xy_plane_vx.png')

            plot_three_planes(vx, Lx, Ly, Lz, t, t0, V_P, V_S, 'Lebedev_three_planes_vx.png', None,'seismic', False)
            plot_three_planes(vz, Lx, Ly, Lz, t, t0, V_P, V_S, 'Lebedev_three_planes_vz.png', None,'seismic', False)

        # if (n == Nt-1):
        #     field = extract_field(v[:sum(blocks.size for blocks in layout_plot.v_blocks.values())], v0_block)
        #     ix = int(0.3 * field.shape[0])
        #     iy = int(0.3 * field.shape[1])
        #     iz = int(0.4 * field.shape[2])
          
        #     plot_cross_section(field, Lx, Ly, Lz, 'x', iz, n)



    
        #PPW
        field = extract_field(v[:sum(blocks.size for blocks in layout_plot.v_blocks.values())], v0_block) #updated field
        ix = int(0.3 * field.shape[0])
        iy = int(0.3 * field.shape[1])
        iz = int(0.4 * field.shape[2])
        trace[n] = field[ix, iy, iz]


 
        # if (n == 1300): #or (n == 150) or (n == 300) or (n == 400) or (n == 500):
        #     v_x = extract_field(v, layout.v_blocks['x'])
        #     plot_cross_section(v_x, Lx, Lz, 'x', iz,  n, title=f'vx slice at n={n}')
       


    writer.close()
    # i0 = Nt // 5      # skip early time
    # i1 = Nt // 2      # before reflections
    i0 = int(0.2 * Nt)
    i1 = int(0.8 * Nt)

    trace_win = trace[i0:i1] #restrict time window 
    
    #make_gif(v_series, sigma_series, layout, Lx, Ly, Lz, '3d_vx.gif', 'vx', None)
    #PPW(trace_win, dt, dx, V_P, V_S, 0.1)

    # plt.plot(norm_hist / norm_hist[10])
    # plt.xlabel('n')
    # plt.ylabel('Norm')
    # plt.show()

    return v, sigma, layouts




def isosurface(v, sigma, layout, Lx, Ly, Lz, field, iso, opacity, show_edges):
    #extract fields
    vx = extract_field(v, layout.v_blocks['x'])
    vy = extract_field(v, layout.v_blocks['y'])
    vz = extract_field(v, layout.v_blocks['z'])

    sxx = extract_field(sigma, layout.sigma_blocks['xx'])
    syy = extract_field(sigma, layout.sigma_blocks['yy'])
    szz = extract_field(sigma, layout.sigma_blocks['zz'])

    if field == 'velmag': #velocity magnitude
        F = np.sqrt(vx**2 + vy**2 + vz**2)
        name = 'velmag'
    elif field == 'pressure':
        F = -(sxx + syy + szz)/3
        name = 'pressure'
    elif field == 'vx':
        F = vx
        name = 'vx'
    elif field == 'vy':
        F = vy
        name = 'vy'
    elif field == 'vz':
        F = vz
        name = 'vz'
    else:
        raise ValueError('field does not exist')
    #build grid from field
    nx, ny, nz = F.shape
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    dz = Lz / (nz - 1)

    grid = pv.ImageData()
    grid.dimensions = (nx,ny,nz)
    grid.origin = (0.0, 0.0, 0.0)
    grid.spacing = (dx, dy, dz)

    grid.point_data[name] = F.flatten(order='F')

    if iso is None:
        iso = 0.25 * np.max(np.abs(F))
    
    if field == 'velmag':
        contours = [iso]
    else:
        contours = [-iso, iso]

    surf =  grid.contour(isosurfaces = contours, scalars = name)

    #render
    p = pv.Plotter()
    p.add_axes()
    p.add_bounding_box()
    p.add_mesh(
        surf, opacity=opacity, show_edges=show_edges, scalar_bar_args={'title': name}
    )
    p.show_grid()
    p.show()
def make_surface(v, sigma, layout, Lx, Ly, Lz, field, iso):
    #extract fields
    vx = extract_field(v, layout.v_blocks['x'])
    vy = extract_field(v, layout.v_blocks['y'])
    vz = extract_field(v, layout.v_blocks['z'])

    sxx = extract_field(sigma, layout.sigma_blocks['xx'])
    syy = extract_field(sigma, layout.sigma_blocks['yy'])
    szz = extract_field(sigma, layout.sigma_blocks['zz'])

    if field == 'velmag': #velocity magnitude
        F = np.sqrt(vx**2 + vy**2 + vz**2)
        name = 'velmag'
    elif field == 'pressure':
        F = -(sxx + syy + szz)/3
        name = 'pressure'
    elif field == 'vx':
        F = vx
        name = 'vx'
    elif field == 'vy':
        F = vy
        name = 'vy'
    elif field == 'vz':
        F = vz
        name = 'vz'
    else:
        raise ValueError('field does not exist')
    #build grid from field
    nx, ny, nz = F.shape
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    dz = Lz / (nz - 1)

    grid = pv.ImageData()
    grid.dimensions = (nx,ny,nz)
    grid.origin = (0.0, 0.0, 0.0)
    grid.spacing = (dx, dy, dz)

    grid.point_data[name] = F.flatten(order='F')
    surf = grid.contour(isosurfaces = [-iso, iso], scalars= name)
    return surf
def make_gif(v_series, sigma_series, layout, Lx, Ly, Lz, filename, field, iso):


    #build surface
    v0 = v_series[0]
    sigma = sigma_series[0] #dont use
    vx0 = extract_field(v0, layout.v_blocks['x'])
    vy0 = extract_field(v0, layout.v_blocks['y'])
    vz0 = extract_field(v0, layout.v_blocks['z'])

    F0 = vx0
    iso = 0.2 * np.max(np.abs(F0))
    surf0 = make_surface(v0, sigma, layout, Lx, Ly, Lz, 'vx', iso)

    p = pv.Plotter(off_screen=True)
    p.open_gif(filename)

    p.add_axes()
    p.add_bounding_box()
    #p.show_grid()#bounds=(0,Lx,0,Ly,0,Lz))
    p.camera_position = [
        (1.8*Lx, 1.8*Ly,1.8*Lz),
        (0.5*Lx, 0.5*Ly, 0.5*Lz),
        (0,0,1)
    ]
    actor = p.add_mesh(surf0, opacity = 0.4)
    p.write_frame()

    for v in v_series[1:]:
        surf = make_surface(v, sigma, layout, Lx, Ly, Lz, 'vx', iso)
        actor.mapper.SetInputData(surf)
        p.write_frame()
    p.close()

    

    

def plot_cross_section(field, Lx,  Ly, Lz, axis, idx, n):
    nx, ny, nz = field.shape
    if axis == 'x':
        x = np.linspace(0,Lx,nx)
        plt.plot(x,field[:,idx,idx])
        plt.xlabel(r'$x$', fontsize=18)
        plt.ylabel(r'$v_x$', fontsize=18)
        #plt.title(f'{title} z-index={idx}')
    elif axis == 'y':
        y = np.linspace(0, Ly, ny)
        plt.plot(y,field[:,idx,:])
    elif axis == 'z':
        z = np.linspace(0,Lz,nz)
        plt.plot(z,field[idx,:,:])
        plt.xlabel(r'$z$', fontsize=18)
        plt.ylabel(r'$v_z$', fontsize=18)
        #plt.title(f'{title} x-index={idx}')
    else: 
        raise ValueError('axis must be x or z')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'Lebedev_v_x_cross_section_n{n}.png')
    plt.show()
    
    

def christoffel(theta, rho, C11, C13, C33, C35, C15, C55):

    kx = np.cos(theta)
    kz = np.sin(theta)

  
    #GENERAL CASE
    G11 = C11*kx**2 + 2*C15*kx*kz + C55*kz**2
    G12 = (C13+C55)*kx*kz + C15*kx**2 + C35*kz**2
    G21 = G12
    G22 = C33*kz**2 + 2*C35*kx*kz + C55*kx**2

    G = np.array([[G11, G12],
                [G12, G22]], dtype=float)
    eigs = np.linalg.eigvalsh(G)

    cS = np.sqrt(eigs[0] / rho)
    cP = np.sqrt(eigs[1] / rho) 
    return cS, cP, G
def curves(lam, mu, rho, ntheta, C11, C13, C33, C35, C15, C55):
    theta = np.linspace(0.0, 2*np.pi, ntheta)

    cP = np.zeros_like(theta)
    cS = np.zeros_like(theta)

    for i, th in enumerate(theta):
        cS[i], cP[i], _ = christoffel(th, rho, C11, C13, C33, C35, C15, C55)

    # angle plot
    plt.figure(figsize=(8, 4.5))
    plt.plot(theta, cP, label="P-wave")
    plt.plot(theta, cS, label="S-wave")
    plt.xlabel(r"$\theta$")
    plt.ylabel("phase velocity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('christoffel_angle_plot.png', dpi=200)
    plt.show()

    # polar Christoffel curves
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(theta, cP, label="P-wave")
    ax.plot(theta, cS, label="S-wave")
    ax.set_title("Christoffel phase-velocity curves")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


    xP = cP * np.cos(theta)
    zP = cP * np.sin(theta)
    xS = cS * np.cos(theta)
    zS = cS * np.sin(theta)

    plt.figure(figsize=(6, 6))
    plt.plot(xP, zP, label="qP")
    plt.plot(xS, zS, label="qS")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("Christoffel curves")
    plt.tight_layout()
    plt.savefig('christoffel_curves.png',dpi=200)
    plt.show()

    return theta, cP, cS
    

def extract_plane(field, plane, idx):
    nx, ny, nz = field.shape

    if plane == 'xy':
        if idx is None:
            idx = nz // 2
        return field[:,:, idx]
    elif plane == 'xz':
        if idx is None:
            idx = ny // 2
        return field[:,idx,:]
    elif plane == 'yz':
        if idx is None:
            idx = nx // 2
        return field[idx, :, :]
    else:
        raise ValueError('invalid plane')

def plot_plane(field, Lx, Ly, Lz, plane, idx, title):
    slab = extract_plane(field, plane, idx)

    if plane == 'xy':
        extent = [0, Lx, 0, Ly]
        xlabel, ylabel = 'x', 'y'
    elif plane == 'xz':
        extent = [0, Lx, 0, Lz]
        xlabel, ylabel = 'x', 'z'
    elif plane == 'yz':
        extent = [0, Ly, 0, Lz]
        xlabel, ylabel = 'y', 'z'
    vmax = np.max(np.abs(slab))
    
    plt.figure(figsize=(6,5))
    plt.imshow(
        slab.T,
        origin='lower',
        extent=extent,
        aspect='auto',
        cmap='seismic',
        vmin= -vmax,
        vmax = vmax,
    )
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()



def plot_three_planes(field3, Lx, Ly, Lz, t, t0, V_P, V_S, title, idx, cmap, circles):
    xy = extract_plane(field3, "xy", idx)
    xz = extract_plane(field3, "xz", idx)
    yz = extract_plane(field3, "yz", idx)
    #nx,ny,nz = field3.shape
    x0 = Lx / 2
    y0 = Ly / 2
    z0 = Lz / 2

    vmax = max(np.max(np.abs(xy)), np.max(np.abs(xz)), np.max(np.abs(yz)))
    if vmax == 0:
        vmax = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

  

    T = max(t - t0,0.0)
    im0 = axes[0].imshow(xy.T, origin="lower", extent=[0, Lx, 0, Ly],
                         aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

    if circles:                
        circ_p = Circle((x0, y0), V_P * T, fill=False, linestyle="--", linewidth=2)
        circ_s = Circle((x0, y0), V_S * T, fill=False, linestyle=":", linewidth=2)

        axes[0].add_patch(circ_p)
        axes[0].add_patch(circ_s)

        axes[0].plot(x0, y0, "ko", markersize=4)
        axes[0].text(x0 + 0.01 * Lx, y0 + V_P * T + 0.01 * Ly, "P", fontsize=12)
        axes[0].text(x0 + 0.01 * Lx, y0 + V_S * T + 0.01 * Ly, "S", fontsize=12)


    axes[0].set_title("xy")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    axes[1].imshow(xz.T, origin="lower", extent=[0, Lx, 0, Lz],
                   aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

    if circles:
        circ_p = Circle((x0, z0), V_P * T, fill=False, linestyle="--", linewidth=2)
        circ_s = Circle((x0, z0), V_S * T, fill=False, linestyle=":", linewidth=2)

        axes[1].add_patch(circ_p)
        axes[1].add_patch(circ_s)

        axes[1].plot(x0, z0, "ko", markersize=4)
        axes[1].text(x0 + 0.01 * Lx, z0 + V_P * T + 0.01 * Lz, "P", fontsize=12)
        axes[1].text(x0 + 0.01 * Lx, z0 + V_S * T + 0.01 * Lz, "S", fontsize=12)

    axes[1].set_title("xz")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")




    axes[2].imshow(yz.T, origin="lower", extent=[0, Ly, 0, Lz],
                   aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    axes[2].set_title("yz")
    axes[2].set_xlabel("y")
    axes[2].set_ylabel("z")
    
    
    if circles:
            
        circ_p = Circle((y0, z0), V_P * T, fill=False, linestyle="--", linewidth=2)
        circ_s = Circle((y0, z0), V_S * T, fill=False, linestyle=":", linewidth=2)
        axes[2].add_patch(circ_p)
        axes[2].add_patch(circ_s)

        axes[2].plot(y0, z0, "ko", markersize=4)
        axes[2].text(y0 + 0.01 * Ly, z0 + V_P * T + 0.01 * Lz, "P", fontsize=12)
        axes[2].text(y0 + 0.01 * Ly, z0 + V_S * T + 0.01 * Lz, "S", fontsize=12)
    

    axes[0].set_xlim(0, Lx); axes[0].set_ylim(0, Ly)
    axes[1].set_xlim(0, Lx); axes[1].set_ylim(0, Lz)
    axes[2].set_xlim(0, Ly); axes[2].set_ylim(0, Lz)
    fig.colorbar(im0, ax=axes, shrink=0.8)
    fig.suptitle(title)
    #plt.tight_layout()
    plt.savefig(f'{title}.png')
    plt.show()






def main():

    #STRESS-VELOCITY CLUSTER PAIRS
    SCL_lst = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    VCL_lst = [(1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]


    # domain and grid
    Lx, Ly, Lz = 1.0, 1.0, 1.0
    Nx, Ny, Nz = 30, 30, 30


    # build layout once to size global state vectors correctly
    # layout = build_layout_from_clusters(SCL[0], VCL[0], Nx, Ny, Nz)

    # N_sigma = sum(block.size for block in layout.sigma_blocks.values())
    # N_v = sum(block.size for block in layout.v_blocks.values())

    #material
    rho = 2.5
    V_P = 3.0
    V_S = 1.0

    mu = rho * V_S**2
    lam = rho * (V_P**2 -2*V_S**2)

    #isotropic case: relation to Lamé parameters
    #C11=C22=C33= lam + 2*mu
    #C12=C13=C23 = lam
    #C44=C55=66= mu

    # C11 = lam+2*mu
    # # C22 = C11
    # # C33 = C11

    # C12 = lam
    # # C13 = C12
    # # C23 = C12
  
    # C44 = mu
    # # C55 = C44
    # # C66 = C44

    # C_mat = np.array([
    #     [C11, C12, C12, 0, 0, 0],
    #     [C12, C11, C12, 0, 0, 0],
    #     [C12, C12, C11, 0, 0, 0],
    #     [0, 0, 0, C44, 0, 0],
    #     [0, 0, 0, 0, C44, 0],
    #     [0, 0, 0, 0, 0, C44]
    # ])


    # #VTI - Thomsen param
    epsilon = 0.334
    gamma = 0.575
    delta_star = 0.93

    
    C33 = rho * V_P**2
    C44 = rho * V_S**2

    C11 = C33 * (1 + 2*epsilon)
    C66 =  C44 * (1 + 2 * gamma)
    C12 = C11 - 2 * C66
    C13 = C33 *delta_star - C44



    # #TTI - rotate stiffness tensor
    # C_mat = np.array([
    # [10, 3, 2, 1, 0.5, 0],
    # [3, 10, 2, 0, 1, 0.5],
    # [2, 2, 8, 0.5, 0, 1],
    # [1, 0, 0.5, 4, 0, 0],
    # [0.5, 1, 0, 0, 4, 0],
    # [0, 0.5, 1, 0, 0, 3]
    # ])


    C_mat = np.array([
        [C11, C12, C13, 0, 0, 0],
        [C12, C11, C13, 0, 0, 0],
        [C13, C13, C33, 0, 0, 0],
        [0, 0, 0, C44, 0, 0],
        [0, 0, 0, 0, C44, 0],
        [0, 0, 0, 0, 0, C66]
    ])
    print(f'Eigvals of C: {np.linalg.eigvalsh(C_mat)}')
    max_eigval_C = np.max(np.linalg.eigvalsh(C_mat))
    V_max = np.sqrt(max_eigval_C / rho)
    dx = Lx / (Nx - 1)
   
    c = 1
    dt = c * dx / (np.sqrt(6)*V_max)
    print(f'dt = {dt}')

    T = 1 #final time
    Nt = int(np.ceil(T / dt)) #number of time steps

    #state vectors
    v = None
    sigma = None

    #initial condition
    #sigma = initial_condition_sigma(sigma, layout, Lx, Ly, Lz, Nx, Ny, Nz, A=1.0, s=0.05)

    #run
    n_plot = 10
    field_gif = "v_x"
    v, sigma, layouts = leapfrog(v, sigma, Lx, Ly, Lz, dt, Nt, Nx, Ny, Nz, rho, C_mat, V_P, V_S, n_plot, field_gif, SCL_lst, VCL_lst)

    layout0 = layouts[0]

    Ns0 = sum(b.size for b in layout0.sigma_blocks.values())
    Nv0 = sum(b.size for b in layout0.v_blocks.values())

    sigma0 = sigma[:Ns0]
    v0 = v[:Nv0]

    sigma_xx = extract_field(sigma0, layout0.sigma_blocks["xx"])
    v_x = extract_field(v0, layout0.v_blocks["x"])



    #extract fields
    # sigma_xx = extract_field(sigma, layout.sigma_blocks["xx"])
    # sigma_yy = extract_field(sigma, layout.sigma_blocks["yy"])
    # sigma_zz = extract_field(sigma, layout.sigma_blocks["zz"])

    # sigma_yz = extract_field(sigma, layout.sigma_blocks["yz"])
    # sigma_xz = extract_field(sigma, layout.sigma_blocks["xz"])
    # sigma_xy = extract_field(sigma, layout.sigma_blocks["xy"])

    # v_x = extract_field(v, layout.v_blocks["x"])
    # v_y = extract_field(v, layout.v_blocks["y"])
    # v_z = extract_field(v, layout.v_blocks["z"])

   
    print(f"Simulating with V_P={V_P}, V_S={V_S}")
    print(f"dt={dt}, Nt={Nt}, Nx={Nx}, Ny={Ny}, Nz={Nz}, h={dx}")
    print("Done.")

    # print("sigma_xx shape:", sigma_xx.shape, "grid:", layout.sigma_fields["xx"].grid)
    # print("sigma_yy shape:", sigma_yy.shape, "grid:", layout.sigma_fields["yy"].grid)
    # print("sigma_zz shape:", sigma_zz.shape, "grid:", layout.sigma_fields["zz"].grid)
    # print("sigma_yz shape:", sigma_yz.shape, "grid:", layout.sigma_fields["yz"].grid)
    # print("sigma_xz shape:", sigma_xz.shape, "grid:", layout.sigma_fields["xz"].grid)
    # print("sigma_xy shape:", sigma_xy.shape, "grid:", layout.sigma_fields["xy"].grid)
    # print("v_x shape:", v_x.shape, "grid:", layout.v_fields["x"].grid)
    # print("v_y shape:", v_y.shape, "grid:", layout.v_fields["y"].grid)
    # print("v_z shape:", v_z.shape, "grid:", layout.v_fields["z"].grid)


    #print(f'Eigvals of C: {np.linalg.eigvalsh(C_mat)}')

    #curves(lam,mu,rho,721, C11, C13, C33, C35, C15, C55)

if __name__ == "__main__":
    main()