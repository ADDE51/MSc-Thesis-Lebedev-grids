import numpy as np
import scipy
from dataclasses import dataclass
from typing import Tuple, Dict
import matplotlib.pyplot as plt

import imageio.v2 as imageio


from scipy.sparse import kron, csc_matrix, eye, bmat, diags, block_diag


# ============================================================
# Full 2D Lebedev elastic wave simulation in the xz-plane
# Unknowns per cluster pair:
#   stress:  (sxx, szz, sxz)
#   velocity:(vx, vz)
#
# Full Lebedev assembly uses two cluster pairs:
#   SCL = (0,0,0), VCL = (1,1,1)
#   SCL = (1,0,1), VCL = (0,1,0)
# ============================================================


# ------------------------------------------------------------
# Binary logic for clusters / grids
# ------------------------------------------------------------

Bit3 = Tuple[int, int, int]


def xor_bit(a: Bit3, b: Bit3) -> Bit3:
    return (a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2])


VELOCITY_SHIFT: Dict[str, Bit3] = {
    "x": (1, 0, 0),
    "z": (0, 0, 1),
}

STRESS_SHIFT: Dict[str, Bit3] = {
    "xx": (1, 1, 1),
    "zz": (1, 1, 1),
    "xz": (0, 1, 0),
}


@dataclass(frozen=True)
class FieldID:
    family: str        # "v" or "sigma"
    component: str     # "x", "z", "xx", "zz", "xz"
    cluster: Bit3

    @property
    def shift(self) -> Bit3:
        if self.family == "v":
            return VELOCITY_SHIFT[self.component]
        if self.family == "sigma":
            return STRESS_SHIFT[self.component]
        raise ValueError(f"Unknown family: {self.family}")

    @property
    def grid(self) -> Bit3:
        return xor_bit(self.cluster, self.shift)

    @property
    def dual_cluster(self) -> Bit3:
        return xor_bit(self.cluster, (1, 1, 1))


@dataclass(frozen=True)
class Block:
    offset: int
    size: int
    shape: tuple[int, int]


@dataclass(frozen=True)
class Layout2D:
    sigma_fields: dict
    v_fields: dict
    sigma_blocks: dict
    v_blocks: dict


# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------

def shape_from_grid_2d(grid: Bit3, Nx: int, Nz: int):
    gx, _, gz = grid
    nx = Nx if gx == 0 else Nx - 1
    nz = Nz if gz == 0 else Nz - 1
    return (nx, nz), nx * nz


def build_layout_from_clusters_2d(SCL: Bit3, VCL: Bit3, Nx: int, Nz: int) -> Layout2D:
    sigma_fields = {
        "xx": FieldID("sigma", "xx", SCL),
        "zz": FieldID("sigma", "zz", SCL),
        "xz": FieldID("sigma", "xz", SCL),
    }
    v_fields = {
        "x": FieldID("v", "x", VCL),
        "z": FieldID("v", "z", VCL),
    }

    sigma_blocks = {}
    offset = 0
    for key in ["xx", "zz", "xz"]:
        shape, size = shape_from_grid_2d(sigma_fields[key].grid, Nx, Nz)
        sigma_blocks[key] = Block(offset, size, shape)
        offset += size

    v_blocks = {}
    offset = 0
    for key in ["x", "z"]:
        shape, size = shape_from_grid_2d(v_fields[key].grid, Nx, Nz)
        v_blocks[key] = Block(offset, size, shape)
        offset += size

    return Layout2D(
        sigma_fields=sigma_fields,
        v_fields=v_fields,
        sigma_blocks=sigma_blocks,
        v_blocks=v_blocks,
    )


def extract_field_2d(vec: np.ndarray, block: Block) -> np.ndarray:
    sl = slice(block.offset, block.offset + block.size)
    return vec[sl].reshape(block.shape, order="F")


# ------------------------------------------------------------
# 1D derivative operators
# ------------------------------------------------------------

def bidiag(N: int):
    interior = [-np.ones(N - 1), np.ones(N - 1)]
    offsets = [0, 1]
    return diags(interior, offsets, shape=(N - 1, N), format="csc")


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


# ------------------------------------------------------------
# 2D derivative expansion
# ------------------------------------------------------------

def derivative_2d(field: FieldID, axis: str, Nx: int, Nz: int, hx_p, hx_d, hz_p, hz_d):
    X_p, X_d = diff_ops_1D(Nx, hx_p, hx_d)
    Z_p, Z_d = diff_ops_1D(Nz, hz_p, hz_d)

    Ix_p = eye(Nx, format="csc")
    Iz_p = eye(Nz, format="csc")
    Ix_d = eye(Nx - 1, format="csc")
    Iz_d = eye(Nz - 1, format="csc")

    gx, _, gz = field.grid

    if axis == "x":
        Dx = X_p if gx == 0 else X_d
        Iz = Iz_p if gz == 0 else Iz_d
        return kron(Iz, Dx, format="csc")

    if axis == "z":
        Dz = Z_p if gz == 0 else Z_d
        Ix = Ix_p if gx == 0 else Ix_d
        return kron(Dz, Ix, format="csc")

    raise ValueError("axis must be 'x' or 'z'")


# ------------------------------------------------------------
# Single cluster-pair operators
# ------------------------------------------------------------

def block_diff_ops_2d(SCL: Bit3, VCL: Bit3, Nx: int, Nz: int, hx_p, hx_d, hz_p, hz_d):
    layout = build_layout_from_clusters_2d(SCL, VCL, Nx, Nz)

    sigma_fields = layout.sigma_fields
    v_fields = layout.v_fields

    # D_sigma: stress divergence into velocity equations
    Dx_sxx = derivative_2d(sigma_fields["xx"], "x", Nx, Nz, hx_p, hx_d, hz_p, hz_d)
    Dz_szz = derivative_2d(sigma_fields["zz"], "z", Nx, Nz, hx_p, hx_d, hz_p, hz_d)
    Dz_sxz = derivative_2d(sigma_fields["xz"], "z", Nx, Nz, hx_p, hx_d, hz_p, hz_d)
    Dx_sxz = derivative_2d(sigma_fields["xz"], "x", Nx, Nz, hx_p, hx_d, hz_p, hz_d)

    D_sigma = bmat([
        [Dx_sxx, None,   Dz_sxz],
        [None,   Dz_szz, Dx_sxz],
    ], format="csc")

    # D_v: velocity gradients into strain-rate vector
    Dx_vx = derivative_2d(v_fields["x"], "x", Nx, Nz, hx_p, hx_d, hz_p, hz_d)
    Dz_vz = derivative_2d(v_fields["z"], "z", Nx, Nz, hx_p, hx_d, hz_p, hz_d)
    Dz_vx = derivative_2d(v_fields["x"], "z", Nx, Nz, hx_p, hx_d, hz_p, hz_d)
    Dx_vz = derivative_2d(v_fields["z"], "x", Nx, Nz, hx_p, hx_d, hz_p, hz_d)

    D_v = bmat([
        [Dx_vx, None],
        [None,  Dz_vz],
        [Dz_vx, Dx_vz],
    ], format="csc")

    return D_sigma, D_v, layout


# ------------------------------------------------------------
# Full 2D Lebedev assembly
# ------------------------------------------------------------

def lebedev_derivative_operators_2d(SCL_lst, VCL_lst, Nx, Nz, hx_p, hx_d, hz_p, hz_d):
    D_s_list = []
    D_v_list = []
    layouts = []

    for scl, vcl in zip(SCL_lst, VCL_lst):
        Ds, Dv, layout = block_diff_ops_2d(scl, vcl, Nx, Nz, hx_p, hx_d, hz_p, hz_d)
        D_s_list.append(Ds)
        D_v_list.append(Dv)
        layouts.append(layout)

    D_sigma = block_diag(D_s_list, format="csc")
    D_v = block_diag(D_v_list, format="csc")
    return D_sigma, D_v, layouts


# ------------------------------------------------------------
# Permutations: cluster ordering -> grid ordering
# 2D analog of your 3D routine
# Stress grids:
#   cluster (000): xx,zz on [111], xz on [010]
#   cluster (101): xx,zz on [010], xz on [111]
#
# Velocity grids:
#   cluster (111): vx,vz on [011], [110]
#   cluster (010): vx,vz on [110], [011]
# ------------------------------------------------------------

def grid_cluster_permutation_2d(layouts):
    # Cluster ordering:
    #   layouts[0] = SCL (000), VCL (111)
    #   layouts[1] = SCL (101), VCL (010)
    #
    # Grid ordering target:
    #   stress blocks grouped by [111], then [010]
    #   velocity blocks grouped by [011], then [110]

    stress_blocks = []
    vel_blocks = []

    for layout in layouts:
        for comp in ["xx", "zz", "xz"]:
            stress_blocks.append(layout.sigma_blocks[comp])
        for comp in ["x", "z"]:
            vel_blocks.append(layout.v_blocks[comp])

    # stress blocks in cluster order:
    # 0:(000)-xx -> [111]
    # 1:(000)-zz -> [111]
    # 2:(000)-xz -> [010]
    # 3:(101)-xx -> [010]
    # 4:(101)-zz -> [010]
    # 5:(101)-xz -> [111]
    #
    # desired grid order:
    # [111]: xx(000), zz(000), xz(101)  => 0,1,5
    # [010]: xz(000), xx(101), zz(101)  => 2,3,4
    idx_s = [0, 1, 5, 2, 3, 4]

    # velocity blocks in cluster order:
    # 0:(111)-vx -> [011]
    # 1:(111)-vz -> [110]
    # 2:(010)-vx -> [110]
    # 3:(010)-vz -> [011]
    #
    # desired grid order:
    # [011]: vx(111), vz(010) => 0,3
    # [110]: vz(111), vx(010) => 1,2
    idx_v = [0, 3, 1, 2]

    n = len(stress_blocks)
    m = len(vel_blocks)

    Ps_blocks = [[None for _ in range(n)] for _ in range(n)]
    Pv_blocks = [[None for _ in range(m)] for _ in range(m)]

    for i in range(n):
        j = idx_s[i]
        bj = stress_blocks[j]
        Ps_blocks[i][j] = eye(bj.size, format="csc")
    P_sigma = bmat(Ps_blocks, format="csc")

    for i in range(m):
        j = idx_v[i]
        bj = vel_blocks[j]
        Pv_blocks[i][j] = eye(bj.size, format="csc")
    P_v = bmat(Pv_blocks, format="csc")

    return P_sigma, P_v


# ------------------------------------------------------------
# Material matrices
# plane strain isotropic: [sxx, szz, sxz]
# ------------------------------------------------------------

def material_matrix_2d(layout: Layout2D, rho: float, lam: float, mu: float):
    N_vx = layout.v_blocks["x"].size
    N_vz = layout.v_blocks["z"].size

    N_sxx = layout.sigma_blocks["xx"].size
    N_szz = layout.sigma_blocks["zz"].size
    N_sxz = layout.sigma_blocks["xz"].size

    M_rho = block_diag((
        rho * eye(N_vx, format="csc"),
        rho * eye(N_vz, format="csc"),
    ), format="csc")

    C11 = lam + 2.0 * mu
    C13 = lam
    C55 = mu

    C11_blk = C11 * eye(N_sxx, format="csc")
    C13_blk = C13 * eye(N_sxx, format="csc")
    C33_blk = C11 * eye(N_szz, format="csc")
    C55_blk = C55 * eye(N_sxz, format="csc")

    C = bmat([
        [C11_blk, C13_blk, None],
        [C13_blk, C33_blk, None],
        [None,    None,    C55_blk],
    ], format="csc")

    return M_rho, C






# def VTI_const_medium_lebedev_material_matrix_2d(layouts, Nx, Nz, rho, C_mat):
#     C11, C12, C22, C33 = C_mat[0,0], C_mat[0,1], C_mat[1,1], C_mat[2,2]
#     # one local plane-strain stiffness block
#     C_loc = np.array([
#         [C11, C12, 0.0],
#         [C12, C22, 0.0],
#         [0.0, 0.0, C33],
#     ], dtype=float)

#     C_loc = csc_matrix(C_loc)

#     # grid sizes from first layout
#     layout0 = layouts[0]

#     # stress grid sizes
#     N111 = layout0.sigma_blocks["xx"].size   # xx,zz on [111]
#     N010 = layout0.sigma_blocks["xz"].size   # xz on [010]

#     # C_hat grouped by cluster ordering, matching D_v block_diag order
#     C_hat = block_diag([
#         kron(C_loc, eye(N111, format="csc")),  # cluster (000)
#         kron(C_loc, eye(N010, format="csc")),  # cluster (101)
#     ], format="csc")

#     # velocity grid sizes
#     _, N011 = shape_from_grid_2d((0, 1, 1), Nx, Nz)
#     _, N110 = shape_from_grid_2d((1, 1, 0), Nx, Nz)

#     Mrho_inv_hat = block_diag([
#         (1.0 / rho) * eye(2 * N011, format="csc"),  # cluster (111)
#         (1.0 / rho) * eye(2 * N110, format="csc"),  # cluster (010)
#     ], format="csc")

#     return C_hat, Mrho_inv_hat


def const_medium_lebedev_material_matrix_2d(layouts, rho, C_mat):
    C_blocks = []
    Mrho_inv_blocks = []
    C11, C12, C22, C33 = C_mat[0,0], C_mat[0,1], C_mat[1,1], C_mat[2,2]
    # one local plane-strain stiffness block
    C_loc = csc_matrix(np.array([
        [C11, C12, 0.0],
        [C12, C22, 0.0],
        [0.0, 0.0, C33],
    ], dtype=float))

    for layout in layouts:
        N_sxx = layout.sigma_blocks["xx"].size
        N_szz = layout.sigma_blocks["zz"].size
        N_sxz = layout.sigma_blocks["xz"].size

        # build local stress constitutive operator with correct block sizes
        C_local = bmat([
            [C11 * eye(N_sxx, format="csc"), C12 * eye(N_sxx, format="csc"), None],
            [C12 * eye(N_szz, format="csc"), C22 * eye(N_szz, format="csc"), None],
            [None, None, C33 * eye(N_sxz, format="csc")],
        ], format="csc")

        N_vx = layout.v_blocks["x"].size
        N_vz = layout.v_blocks["z"].size
        Mrho_inv_local = block_diag((
            (1.0 / rho) * eye(N_vx, format="csc"),
            (1.0 / rho) * eye(N_vz, format="csc"),
        ), format="csc")

        C_blocks.append(C_local)
        Mrho_inv_blocks.append(Mrho_inv_local)

    C_hat = block_diag(C_blocks, format="csc")
    Mrho_inv_hat = block_diag(Mrho_inv_blocks, format="csc")
    return C_hat, Mrho_inv_hat

# def iso_const_medium_lebedev_material_matrix_2d(layouts, Nx, Nz, rho, lam, mu):
#     # one local plane-strain stiffness block
#     C_loc = np.array([
#         [lam + 2.0 * mu, lam, 0.0],
#         [lam, lam + 2.0 * mu, 0.0],
#         [0.0, 0.0, mu],
#     ], dtype=float)

#     C_loc = csc_matrix(C_loc)

#     # grid sizes from first layout
#     layout0 = layouts[0]

#     # stress grid sizes
#     N111 = layout0.sigma_blocks["xx"].size   # xx,zz on [111]
#     N010 = layout0.sigma_blocks["xz"].size   # xz on [010]

#     # C_hat grouped by cluster ordering, matching D_v block_diag order
#     C_hat = block_diag([
#         kron(C_loc, eye(N111, format="csc")),  # cluster (000)
#         kron(C_loc, eye(N010, format="csc")),  # cluster (101)
#     ], format="csc")

#     # velocity grid sizes
#     _, N011 = shape_from_grid_2d((0, 1, 1), Nx, Nz)
#     _, N110 = shape_from_grid_2d((1, 1, 0), Nx, Nz)

#     Mrho_inv_hat = block_diag([
#         (1.0 / rho) * eye(2 * N011, format="csc"),  # cluster (111)
#         (1.0 / rho) * eye(2 * N110, format="csc"),  # cluster (010)
#     ], format="csc")

#     return C_hat, Mrho_inv_hat


def plot_velocity_pair_2d(v_cluster, layout: Layout2D, Lx: float, Lz: float,
                          title: str = "Velocity fields",
                          savepath: str = None):
    """
    Plot vx and vz from one cluster side by side.
    """
    vx = extract_field_2d(v_cluster, layout.v_blocks["x"])
    vz = extract_field_2d(v_cluster, layout.v_blocks["z"])

    vmax = max(np.max(np.abs(vx)), np.max(np.abs(vz)))
    if vmax == 0:
        vmax = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    im0 = axes[0].imshow(
        vx.T, origin="lower", extent=[0, Lx, 0, Lz],
        aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax
    )
    axes[0].set_title(r"$v_x$")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("z")

    axes[1].imshow(
        vz.T, origin="lower", extent=[0, Lx, 0, Lz],
        aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax
    )
    axes[1].set_title(r"$v_z$")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")

    fig.colorbar(im0, ax=axes, shrink=0.85)
    fig.suptitle(title)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200)

    plt.show()

# ------------------------------------------------------------
# Source
# ------------------------------------------------------------

def gaussian_on_block_grid_2d(block: Block, Lx, Lz, x0, z0, s):
    x = np.linspace(0, Lx, block.shape[0])
    z = np.linspace(0, Lz, block.shape[1])
    X, Z = np.meshgrid(x, z, indexing="ij")
    g = np.exp(-((X - x0) ** 2 + (Z - z0) ** 2) / (2 * s ** 2))
    return g.flatten(order="F")


def source_2d(layout: Layout2D, Lx, Lz, x0, z0, s, amp):
    N_sigma = sum(block.size for block in layout.sigma_blocks.values())
    q_sigma = np.zeros(N_sigma)

    # inject in sxz
    g_xz = gaussian_on_block_grid_2d(layout.sigma_blocks["xz"], Lx, Lz, x0, z0, s)
    g_xx = gaussian_on_block_grid_2d(layout.sigma_blocks["xx"], Lx, Lz, x0, z0, s)
    g_zz = gaussian_on_block_grid_2d(layout.sigma_blocks["zz"], Lx, Lz, x0, z0, s)

    xz_block = layout.sigma_blocks["xz"]
    xx_block = layout.sigma_blocks["xx"]
    zz_block = layout.sigma_blocks["zz"]

    #q_sigma[xz_block.offset:xz_block.offset + xz_block.size] = amp * g_xz
    q_sigma[xx_block.offset:xx_block.offset + xx_block.size] = amp * g_xx
    q_sigma[zz_block.offset:zz_block.offset + zz_block.size] = amp * g_zz
    return q_sigma


def ricker(t, f0, t0):
    a = np.pi * f0 * (t - t0)
    return (1.0 - 2.0 * a ** 2) * np.exp(-a ** 2)


# ------------------------------------------------------------
# Leapfrog for full 2D Lebedev scheme
# ------------------------------------------------------------



def make_gif_2d(field_series, Lx, Lz, filename,
                cmap="seismic", fps=15, title_prefix=""):
    """
    Build a GIF from a list of 2D numpy arrays.
    All frames use the same color scale.
    """
    vmax = max(np.max(np.abs(F)) for F in field_series)
    if vmax == 0:
        vmax = 1.0

    with imageio.get_writer(filename, mode="I", fps=fps) as writer:
        for k, F in enumerate(field_series):
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(
                F.T,
                origin="lower",
                extent=[0, Lx, 0, Lz],
                aspect="auto",
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
            )
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            ax.set_title(f"{title_prefix} frame {k}")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()

            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            writer.append_data(frame[:, :, :3])  # drop alpha
            plt.close(fig)


def cluster_sizes(layout: Layout2D): #function that comuputes sizes of cluster, #DOFS
    Ns = sum(block.size for block in layout.sigma_blocks.values())
    Nv = sum(block.size for block in layout.v_blocks.values())
    return Ns, Nv


def extract_cluster_state(v_full, sigma_full, layouts, cluster_id: int): #extracts velocity/stress subvectors for one cluster pair from the full Lebedev vectors.
   
    sigma_offsets = []
    v_offsets = []

    s0 = 0
    v0 = 0
    for layout in layouts: #two clusters
        Ns, Nv = cluster_sizes(layout)
        sigma_offsets.append((s0, s0 + Ns))
        v_offsets.append((v0, v0 + Nv))
        s0 += Ns
        v0 += Nv

    sa, sb = sigma_offsets[cluster_id]
    va, vb = v_offsets[cluster_id]

    return v_full[va:vb], sigma_full[sa:sb], layouts[cluster_id]

def extract_named_field_2d(v_full, sigma_full, layouts, cluster_id, family, component):
    """
    Extract a 2D field from one cluster of the full Lebedev vectors.
    family = 'v' or 'sigma'
    component:
      velocity -> 'x', 'z'
      stress   -> 'xx', 'zz', 'xz'
    """
    v_loc, sigma_loc, layout = extract_cluster_state(v_full, sigma_full, layouts, cluster_id)

    if family == "v":
        return extract_field_2d(v_loc, layout.v_blocks[component])
    elif family == "sigma":
        return extract_field_2d(sigma_loc, layout.sigma_blocks[component])
    else:
        raise ValueError("family must be 'v' or 'sigma'")





def debug_permutation(layouts):
    print("\n--- STRESS BLOCKS IN CLUSTER ORDER ---")
    stress_meta = []
    for li, layout in enumerate(layouts):
        for comp in ["xx", "zz", "xz"]:
            field = layout.sigma_fields[comp]
            block = layout.sigma_blocks[comp]
            stress_meta.append({
                "layout_id": li,
                "component": comp,
                "grid": field.grid,
                "size": block.size,
                "shape": block.shape,
            })

    for i, m in enumerate(stress_meta):
        print(
            f"s[{i}] : layout={m['layout_id']} comp={m['component']} "
            f"grid={m['grid']} size={m['size']} shape={m['shape']}"
        )

    idx_s = [0, 1, 5, 2, 3, 4]
    print("\n--- STRESS BLOCKS IN PERMUTED / GRID ORDER ---")
    for i, j in enumerate(idx_s):
        m = stress_meta[j]
        print(
            f"s_perm[{i}] <- s[{j}] : layout={m['layout_id']} comp={m['component']} "
            f"grid={m['grid']} size={m['size']} shape={m['shape']}"
        )

    print("\nExpected stress grid grouping:")
    print(" first 3 blocks should all be grid (1,1,1)")
    print(" last  3 blocks should all be grid (0,1,0)")

    print("\n--- VELOCITY BLOCKS IN CLUSTER ORDER ---")
    vel_meta = []
    for li, layout in enumerate(layouts):
        for comp in ["x", "z"]:
            field = layout.v_fields[comp]
            block = layout.v_blocks[comp]
            vel_meta.append({
                "layout_id": li,
                "component": comp,
                "grid": field.grid,
                "size": block.size,
                "shape": block.shape,
            })

    for i, m in enumerate(vel_meta):
        print(
            f"v[{i}] : layout={m['layout_id']} comp={m['component']} "
            f"grid={m['grid']} size={m['size']} shape={m['shape']}"
        )

    idx_v = [0, 3, 1, 2]
    print("\n--- VELOCITY BLOCKS IN PERMUTED / GRID ORDER ---")
    for i, j in enumerate(idx_v):
        m = vel_meta[j]
        print(
            f"v_perm[{i}] <- v[{j}] : layout={m['layout_id']} comp={m['component']} "
            f"grid={m['grid']} size={m['size']} shape={m['shape']}"
        )

    print("\nExpected velocity grid grouping:")
    print(" first 2 blocks should all be grid (0,1,1)")
    print(" last  2 blocks should all be grid (1,1,0)")




def simple_norm(v, sigma):
    return np.sqrt(np.dot(v, v) + np.dot(sigma, sigma))



def Dconst_medium_lebedev_material_matrix_2d(layouts, rho, C_mat):

    L0, L1 = layouts 

    # unpack 2D constitutive coefficients
    C11 = float(C_mat[0, 0])
    C13 = float(C_mat[0, 1])
    C15 = float(C_mat[0, 2])
    C33 = float(C_mat[1, 1])
    C35 = float(C_mat[1, 2])
    C55 = float(C_mat[2, 2])

    # ------------------------------------------------------------
    # Stress constitutive operator in PERMUTED / GRID ORDER
    # ------------------------------------------------------------

    # grid (111): [xx(L0), zz(L0), xz(L1)]
    N_xx_111 = L0.sigma_blocks["xx"].size
    N_zz_111 = L0.sigma_blocks["zz"].size
    N_xz_111 = L1.sigma_blocks["xz"].size

    if not (N_xx_111 == N_zz_111 == N_xz_111):
        raise ValueError(
            f"Grid (111) stress sizes do not match: "
            f"xx(L0)={N_xx_111}, zz(L0)={N_zz_111}, xz(L1)={N_xz_111}"
        )

    I111 = eye(N_xx_111, format="csc")
    C_111 = bmat([
        [C11 * I111, C13 * I111, C15 * I111],
        [C13 * I111, C33 * I111, C35 * I111],
        [C15 * I111, C35 * I111, C55 * I111],
    ], format="csc")

    # grid (010): [xz(L0), xx(L1), zz(L1)]
    N_xz_010 = L0.sigma_blocks["xz"].size
    N_xx_010 = L1.sigma_blocks["xx"].size
    N_zz_010 = L1.sigma_blocks["zz"].size

    if not (N_xz_010 == N_xx_010 == N_zz_010):
        raise ValueError(
            f"Grid (010) stress sizes do not match: "
            f"xz(L0)={N_xz_010}, xx(L1)={N_xx_010}, zz(L1)={N_zz_010}"
        )

    I010 = eye(N_xz_010, format="csc")

    # Important:
    # the constitutive law is still in physical component order [xx, zz, xz].
    # But this grid-group vector is ordered [xz, xx, zz], so we must reorder.
    #
    # Let p = [xz, xx, zz]^T and u = [xx, zz, xz]^T.
    # Then u = R @ p, where
    #   R = [[0,1,0],
    #        [0,0,1],
    #        [1,0,0]]
    #
    # The operator acting on p is: C_p = R.T @ C_phys @ R
    C_phys = np.array([
        [C11, C13, C15],
        [C13, C33, C35],
        [C15, C35, C55],
    ], dtype=float)

    R = np.array([
        [0.0, 1.0, 0.0],  # xx <- entry 1 of p
        [0.0, 0.0, 1.0],  # zz <- entry 2 of p
        [1.0, 0.0, 0.0],  # xz <- entry 0 of p
    ], dtype=float)

    C_010_small = R.T @ C_phys @ R

    C_010 = bmat([
        [C_010_small[0, 0] * I010, C_010_small[0, 1] * I010, C_010_small[0, 2] * I010],
        [C_010_small[1, 0] * I010, C_010_small[1, 1] * I010, C_010_small[1, 2] * I010],
        [C_010_small[2, 0] * I010, C_010_small[2, 1] * I010, C_010_small[2, 2] * I010],
    ], format="csc")

    C_hat = block_diag((C_111, C_010), format="csc")

    # ------------------------------------------------------------
    # Velocity density operator in PERMUTED / GRID ORDER
    # ------------------------------------------------------------

    # grid (011): [vx(L0), vz(L1)]
    N_vx_011 = L0.v_blocks["x"].size
    N_vz_011 = L1.v_blocks["z"].size
    if N_vx_011 != N_vz_011:
        raise ValueError(
            f"Grid (011) velocity sizes do not match: "
            f"vx(L0)={N_vx_011}, vz(L1)={N_vz_011}"
        )

    # grid (110): [vz(L0), vx(L1)]
    N_vz_110 = L0.v_blocks["z"].size
    N_vx_110 = L1.v_blocks["x"].size
    if N_vz_110 != N_vx_110:
        raise ValueError(
            f"Grid (110) velocity sizes do not match: "
            f"vz(L0)={N_vz_110}, vx(L1)={N_vx_110}"
        )

    Mrho_inv_011 = block_diag((
        (1.0 / rho) * eye(N_vx_011, format="csc"),
        (1.0 / rho) * eye(N_vz_011, format="csc"),
    ), format="csc")

    Mrho_inv_110 = block_diag((
        (1.0 / rho) * eye(N_vz_110, format="csc"),
        (1.0 / rho) * eye(N_vx_110, format="csc"),
    ), format="csc")

    Mrho_inv_hat = block_diag((Mrho_inv_011, Mrho_inv_110), format="csc")

    return C_hat, Mrho_inv_hat







def energy(v, sigma, rho, Cinv, h):
    Ev = h**3 * rho * np.dot(v, v)
    Es = h**3 * sigma @ (Cinv @ sigma)
    return Ev + Es




def leapfrog_2d_lebedev(v, sigma, Lx, Lz, dt, Nt, Nx, Nz, rho, C_mat, Vp, Vs,
                            SCL_lst, VCL_lst,
                            gif_filename="lebedev_2d.gif",
                            gif_family="v", gif_component="x",
                            gif_cluster=0, frame_stride=10, fps=15):
    dx = Lx / (Nx - 1)
    dz = Lz / (Nz - 1)

    hx_p = hx_d = dx
    hz_p = hz_d = dz

    D_sigma, D_v, layouts = lebedev_derivative_operators_2d(
        SCL_lst, VCL_lst, Nx, Nz, hx_p, hx_d, hz_p, hz_d
    )


    #debug_permutation(layouts)

    P_sigma, P_v = grid_cluster_permutation_2d(layouts)
    C_hat, Mrho_inv_hat = Dconst_medium_lebedev_material_matrix_2d(layouts, rho, C_mat)




    
    ######### COMPUTE spectral radius of A FOR SMALL GRIDS #########
    N_v = D_v.shape[1]
    N_sigma = D_sigma.shape[1]


    # h = hx_p #constant grid size
    # W_s_root = (1.0  / np.sqrt(h)) * eye(N_sigma)
    # W_v_root = np.sqrt(h) * eye(N_v)

    # M_rho_root = scipy.linalg.sqrtm(Mrho_inv_hat.toarray())
    # C_hat_root = scipy.linalg.sqrtm(C_hat.toarray())
    # B = W_v_root @ M_rho_root @ D_sigma.toarray() @ C_hat_root @ W_s_root
    # R_spec = scipy.linalg.svdvals(B)[0]
    # print(f'rho(B)={R_spec}')
    # print("B shape:", B.shape)
    # print(f'NUM dt <= {2 / R_spec}')

    # Z_v = np.zeros((N_v, N_v))
    # Z_s = np.zeros((N_sigma, N_sigma))

    # A = np.block([
    #     [Z_v, B],
    #     [-B.T, Z_s]
    # ])

    # eigvals = scipy.linalg.eigvals(A)
    # rho_A = np.max(np.abs(eigvals))
    # print(f'rho(A)={rho_A}')


    # import scipy.sparse as sp

    # D = D_sigma.tocsr()
    # G = (D.T @ D).tocsr()   # Gram matrix, shape (N_sigma, N_sigma)

    # diag = G.diagonal()
    # row_sums = np.abs(G).sum(axis=1).A.ravel()
    # gersh_bound = np.max(row_sums)   # since row_sums = a_ii + sum_{j!=i}|a_ij|

    # D_norm_bound = np.sqrt(gersh_bound)

    # print("Gershgorin bound for ||D_sigma||_2:", D_norm_bound)
    # print("Actual ||D_sigma||_2:", np.linalg.svd(D.toarray(), compute_uv=False)[0])
    # abs_row_sums_D = np.abs(D).sum(axis=1).A.ravel()
    # abs_col_sums_D = np.abs(D).sum(axis=0).A.ravel()
    # col_nnz = np.diff(D.tocsc().indptr)

    # print("max row abs sum of D:", abs_row_sums_D.max())
    # print("max col abs sum of D:", abs_col_sums_D.max())
    # print("max nnz per col:", col_nnz.max())



    if v is None or len(v) != N_v:
        v = np.zeros(N_v)
    if sigma is None or len(sigma) != N_sigma:
        sigma = np.zeros(N_sigma)

    q_v = np.zeros(N_v)

    x0, z0 = Lx / 2, Lz / 2
    s = 0.05
    f0 = 30
    t0 = 1.5 / f0

    field_series = []
    norm_hist = []
    for n in range(Nt):
        t = n * dt
        pulse = ricker(t, f0, t0)

        q_sigma = np.concatenate([
            source_2d(layout, Lx, Lz, x0, z0, s, pulse)
            for layout in layouts
        ])

        # sigma^{n+1/2}
        sigma = sigma + dt * (P_sigma.T @ (C_hat @ (P_sigma @ (q_sigma + D_v @ v))))

        # v^{n+1}
        v = v + dt * (P_v.T @ (Mrho_inv_hat @ (P_v @ (q_v + D_sigma @ sigma))))

        #norm_hist.append(energy(v, sigma, rho, C_hat, dx))
        norm_hist.append(simple_norm(v,sigma))
        if n % frame_stride == 0 or n == Nt - 1:
            print(simple_norm(v,sigma))
            F = extract_named_field_2d(
                v, sigma, layouts,
                cluster_id=gif_cluster,
                family=gif_family,
                component=gif_component,
            )
            field_series.append(F.copy())

    make_gif_2d(
        field_series,
        Lx=Lx,
        Lz=Lz,
        filename=gif_filename,
        cmap="seismic",
        fps=fps,
        title_prefix=f"{gif_family}_{gif_component}, cluster {gif_cluster}"
    )

    plt.plot(norm_hist / norm_hist[10])
    plt.show()
    return v, sigma, layouts


def main():
    # Full 2D Lebedev cluster pairs in xz-plane
    SCL_lst = [(0, 0, 0), (1, 0, 1)]
    VCL_lst = [(1, 1, 1), (0, 1, 0)]

    Lx, Lz = 1.0, 1.0
    Nx, Nz = 30, 30

    rho = 2.5
    Vp = 3.0
    Vs = 1.0
    # #VTI
    # C11 = 10
    # C12 = 3.2
    # C22 = 4.0
    # C33 = 5.3

    #ISO
    lam = rho * (Vp ** 2 - 2.0 * Vs ** 2)
    mu = rho * Vs ** 2
    C11 = lam + 2*mu
    C12 = lam
    C22 = C11
    C33 = mu
    C_mat = np.array([
        [C11, C12, 0],
        [C12, C22, 0],
        [0, 0, C33]
    ])

    dx = Lx / (Nx - 1)
    dz = Lz / (Nz - 1)

    # Conservative placeholder CFL
    max_eigval_C = np.max(np.linalg.eigvalsh(C_mat))
    print(f'Max eigval of C: {max_eigval_C}')
    V_max = np.sqrt(max_eigval_C / rho)
    print(f'V_max = {V_max}')

    c = 1.7
    dt = c * dx / (2* V_max) #fails at factor 2.4s
    print(f'ESTIMATE dt = {dt}')



   
    #CFL ESTIMATE
    #dt = 0.25 / (Vp * np.sqrt(1.0 / dx**2 + 1.0 / dz**2))
    T = 1.0
    Nt = int(np.ceil(T / dt))

    v = None
    sigma = None

    v, sigma, layouts = leapfrog_2d_lebedev(
        v=v,
        sigma=sigma,
        Lx=Lx,
        Lz=Lz,
        dt=dt,
        Nt=Nt,
        Nx=Nx,
        Nz=Nz,
        rho=rho,
        C_mat=C_mat,
        Vp=Vp,
        Vs=Vs,
        SCL_lst=SCL_lst,
        VCL_lst=VCL_lst,
        gif_filename="lebedev_2d_vx.gif",
        gif_family="v",
        gif_component="x",
        gif_cluster=0,
        frame_stride=10,
        fps=15,
    )


    layout0 = layouts[0]
    Ns0 = sum(b.size for b in layout0.sigma_blocks.values())
    Nv0 = sum(b.size for b in layout0.v_blocks.values())

    sigma0 = sigma[:Ns0]
    v0 = v[:Nv0]

    vx = extract_field_2d(v0, layout0.v_blocks["x"])
    vz = extract_field_2d(v0, layout0.v_blocks["z"])
    sxx = extract_field_2d(sigma0, layout0.sigma_blocks["xx"])
    sxz = extract_field_2d(sigma0, layout0.sigma_blocks["xz"])

    print("Done.")
    print("dt =", dt, "Nt =", Nt, "h =", dx)
    print('===== CLUSTER (000)-(111) =====')
    print("vx shape:", vx.shape, "grid:", layout0.v_fields["x"].grid)
    print("vz shape:", vz.shape, "grid:", layout0.v_fields["z"].grid)
    print("sxx shape:", sxx.shape, "grid:", layout0.sigma_fields["xx"].grid)
    print("sxz shape:", sxz.shape, "grid:", layout0.sigma_fields["xz"].grid)

    
    Ns1 = sum(b.size for b in layouts[1].sigma_blocks.values())
    Nv1 = sum(b.size for b in layouts[1].v_blocks.values())

    sigma1 = sigma[Ns0: Ns0 + Ns1]
    v1 = v[Nv0: Nv0 + Nv1]

    vx = extract_field_2d(v1, layouts[1].v_blocks["x"])
    vz = extract_field_2d(v1, layouts[1].v_blocks["z"])
    sxx = extract_field_2d(sigma1, layouts[1].sigma_blocks["xx"])
    sxz = extract_field_2d(sigma1, layouts[1].sigma_blocks["xz"])


    print()
    print('===== CLUSTER (101)-(010) =====')
    print("vx shape:", vx.shape, "grid:", layouts[1].v_fields["x"].grid)
    print("vz shape:", vz.shape, "grid:", layouts[1].v_fields["z"].grid)
    print("sxx shape:", sxx.shape, "grid:", layouts[1].sigma_fields["xx"].grid)
    print("sxz shape:", sxz.shape, "grid:", layouts[1].sigma_fields["xz"].grid)


if __name__ == "__main__":
    main()