import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

from scipy.sparse import kron, csc_matrix, eye, bmat, diags, block_diag
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import imageio.v2 as imageio


# ============================================================
# 2D isotropic elastic wave simulation on a single staggered grid
# with cluster/grid logic and block-wise operators
# ============================================================


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
    shape: tuple[int, int]

#this class contains the fields, and their blocks
@dataclass(frozen=True)
class Layout2D:
    sigma_fields: dict
    v_fields: dict
    sigma_blocks: dict
    v_blocks: dict


# ------------------------------------------------------------
# Layout from derived grids
# ------------------------------------------------------------

#function that determines the shape of matrices based on the grid
def shape_from_grid(grid: Bit3, Nx: int, Nz: int):
    gx, _, gz = grid
    nx = Nx if gx == 0 else Nx - 1
    nz = Nz if gz == 0 else Nz - 1
    return (nx, nz), nx * nz

#using stress-velocity cluster, define all variable fields and the stress/vel blocks
def build_layout_from_clusters(SCL: Bit3, VCL: Bit3, Nx: int, Nz: int) -> Layout2D:

    sigma_fields = { #define the 3 stress unknowns with their cluster
        "xx": FieldID("sigma", "xx", SCL),
        "zz": FieldID("sigma", "zz", SCL),
        "xz": FieldID("sigma", "xz", SCL),
    }
    v_fields = { #define the 2 velocity unknowns with their cluster (dual to SCL)
        "x": FieldID("v", "x", VCL),
        "z": FieldID("v", "z", VCL),
    }


    #fill the sigma/velocity blocks with data from the grids
    sigma_blocks = {}
    offset = 0
    for key in ["xx", "zz", "xz"]:
        shape, size = shape_from_grid(sigma_fields[key].grid, Nx, Nz)
        sigma_blocks[key] = Block(offset, size, shape)
        offset += size

    v_blocks = {}
    offset = 0
    for key in ["x", "z"]:
        shape, size = shape_from_grid(v_fields[key].grid, Nx, Nz)
        v_blocks[key] = Block(offset, size, shape)
        offset += size

    return Layout2D( #returns object that contains the fields and blocks for this cluster/grid layout
        sigma_fields=sigma_fields,
        v_fields=v_fields,
        sigma_blocks=sigma_blocks,
        v_blocks=v_blocks,
    )

#function that extracts and reshapes the variable from the block
def extract_field(vec: np.ndarray, block: Block) -> np.ndarray:
    sl = slice(block.offset, block.offset + block.size)
    return vec[sl].reshape(block.shape, order='F')



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
def derivative_2d(field: FieldID, axis: str, Nx: int, Nz: int, hx_p, hx_d, hz_p, hz_d):
    X_p, X_d = diff_ops_1D(Nx, hx_p, hx_d)
    Z_p, Z_d = diff_ops_1D(Nz, hz_p, hz_d)

    #define I matrices for primary and dual (in interior)
    Ix_p = eye(Nx, format="csc")
    Iz_p = eye(Nz, format="csc")
    Ix_d = eye(Nx - 1, format="csc")
    Iz_d = eye(Nz - 1, format="csc")

    gx, _, gz = field.grid 

    #for x or z derivative, depending on the grid, choose diff op and correct sizes for I
    if axis == "x":
        Dx = X_p if gx == 0 else X_d
        Iz = Iz_p if gz == 0 else Iz_d
        return kron(Iz, Dx, format="csc")

    if axis == "z":
        Dz = Z_p if gz == 0 else Z_d
        Ix = Ix_p if gx == 0 else Ix_d
        return kron(Dz, Ix, format="csc")

    raise ValueError("axis must be 'x' or 'z'")


#function that assembles diff ops into block form
def block_diff_ops(SCL: Bit3, VCL: Bit3, Nx: int, Nz: int, hx_p, hx_d, hz_p, hz_d):
    
    layout = build_layout_from_clusters(SCL, VCL, Nx, Nz) #returns Layout2D object

    #extract fields/blocks from Layout2D object
    sigma_fields = layout.sigma_fields
    v_fields = layout.v_fields
    sigma_blocks = layout.sigma_blocks
    v_blocks = layout.v_blocks

    #elements of D_sigma 
    Dx_sxx = derivative_2d(sigma_fields["xx"], "x", Nx, Nz, hx_p, hx_d, hz_p, hz_d)
    Dz_szz = derivative_2d(sigma_fields["zz"], "z", Nx, Nz, hx_p, hx_d, hz_p, hz_d)
    Dz_sxz = derivative_2d(sigma_fields["xz"], "z", Nx, Nz, hx_p, hx_d, hz_p, hz_d)
    Dx_sxz = derivative_2d(sigma_fields["xz"], "x", Nx, Nz, hx_p, hx_d, hz_p, hz_d)

    #zero entries
    Z_vx_sigma_zz = csc_matrix((v_blocks["x"].size, sigma_blocks["zz"].size))
    Z_vz_sigma_xx = csc_matrix((v_blocks["z"].size, sigma_blocks["xx"].size))

    #construct D_sigma
    D_sigma = -bmat([
        [Dx_sxx,        Z_vx_sigma_zz, Dz_sxz],
        [Z_vz_sigma_xx, Dz_szz,        Dx_sxz],
    ], format="csc")

    #elements of D_v
    Dx_vx = derivative_2d(v_fields["x"], "x", Nx, Nz, hx_p, hx_d, hz_p, hz_d)
    Dz_vz = derivative_2d(v_fields["z"], "z", Nx, Nz, hx_p, hx_d, hz_p, hz_d)
    Dz_vx = derivative_2d(v_fields["x"], "z", Nx, Nz, hx_p, hx_d, hz_p, hz_d)
    Dx_vz = derivative_2d(v_fields["z"], "x", Nx, Nz, hx_p, hx_d, hz_p, hz_d)

    #zero entries
    Z_sxx_vz = csc_matrix((sigma_blocks["xx"].size, v_blocks["z"].size))
    Z_szz_vx = csc_matrix((sigma_blocks["zz"].size, v_blocks["x"].size))

    #construct D_v
    D_v = -bmat([
        [Dx_vx,    Z_sxx_vz],
        [Z_szz_vx, Dz_vz],
        [Dz_vx,    Dx_vz],
    ], format="csc")

    return D_sigma, D_v, layout


#function that defines density matrix and stiffness matrix with Lame param
def material_matrix(layout: Layout2D, rho: float, lam: float, mu: float):
    #isotropic homogeneous medium
    N_vx = layout.v_blocks["x"].size
    N_vz = layout.v_blocks["z"].size

    N_sxx = layout.sigma_blocks["xx"].size
    N_szz = layout.sigma_blocks["zz"].size
    N_sxz = layout.sigma_blocks["xz"].size

    M_rho_vx = rho * eye(N_vx, format="csc")
    M_rho_vz = rho * eye(N_vz, format="csc")
    M_rho = block_diag((M_rho_vx, M_rho_vz), format="csc")

    lam_xx = lam * np.ones(N_sxx)
    mu_xx = mu * np.ones(N_sxx)
    lam_zz = lam * np.ones(N_szz)
    mu_zz = mu * np.ones(N_szz)
    mu_xz = mu * np.ones(N_sxz)

    C1 = diags(lam_xx + 2 * mu_xx, format="csc")
    C2 = diags(lam_zz, format="csc")
    C3 = diags(mu_xz, format="csc")

    C = bmat([
        [C1, C2, None],
        [C2, C1, None],
        [None, None, C3],
    ], format="csc")

    return M_rho, C


# ------------------------------------------------------------
# Initial conditions / sources
# ------------------------------------------------------------

def initial_condition_sigma(sigma, layout: Layout2D, Lx, Lz, Nx, Nz, A, s):
    xx_block = layout.sigma_blocks["xx"]
    zz_block = layout.sigma_blocks["zz"]

    xx_shape = xx_block.shape
    zz_shape = zz_block.shape

    x_xx = np.linspace(0, Lx, xx_shape[0])
    z_xx = np.linspace(0, Lz, xx_shape[1])
    X_xx, Z_xx = np.meshgrid(x_xx, z_xx, indexing="ij")

    x_zz = np.linspace(0, Lx, zz_shape[0])
    z_zz = np.linspace(0, Lz, zz_shape[1])
    X_zz, Z_zz = np.meshgrid(x_zz, z_zz, indexing="ij")

    x0 = Lx / 2
    z0 = Lz / 2

    g_xx = A * np.exp(-((X_xx - x0) ** 2 + (Z_xx - z0) ** 2) / (2 * s ** 2))
    g_zz = A * np.exp(-((X_zz - x0) ** 2 + (Z_zz - z0) ** 2) / (2 * s ** 2))

    sigma[xx_block.offset:xx_block.offset + xx_block.size] = g_xx.flatten(order="F")
    sigma[zz_block.offset:zz_block.offset + zz_block.size] = -g_zz.flatten(order="F")
    return sigma


def gaussian_on_block_grid(block: Block, Lx, Lz, x0, z0, s):
    x = np.linspace(0, Lx, block.shape[0])
    z = np.linspace(0, Lz, block.shape[1])
    X, Z = np.meshgrid(x, z, indexing="ij")
    g = np.exp(-((X - x0) ** 2 + (Z - z0) ** 2) / (2 * s ** 2))
    return g.flatten(order="F")


def pressure_source(layout: Layout2D, Lx, Lz, x0, z0, s, amp):
    N_sigma = sum(block.size for block in layout.sigma_blocks.values())
    q_sigma = np.zeros(N_sigma)

    g_xx = gaussian_on_block_grid(layout.sigma_blocks["xx"], Lx, Lz, x0, z0, s)
    g_zz = gaussian_on_block_grid(layout.sigma_blocks["zz"], Lx, Lz, x0, z0, s)

    xx_block = layout.sigma_blocks["xx"]
    zz_block = layout.sigma_blocks["zz"]

    q_sigma[xx_block.offset:xx_block.offset + xx_block.size] = amp * g_xx
    q_sigma[zz_block.offset:zz_block.offset + zz_block.size] = amp * g_zz
    return q_sigma


def ricker(t, f0, t0):
    a = np.pi * f0 * (t - t0)
    return (1.0 - 2.0 * a ** 2) * np.exp(-a ** 2)


# ------------------------------------------------------------
# Plotting / GIF
# ------------------------------------------------------------

def save_snapshot(filename, field_tag, sigma, v, layout: Layout2D, V_P, V_S, Lx, Lz, n, t):
    if field_tag == "sigma_xx":
        field = extract_field(sigma, layout.sigma_blocks["xx"])
    elif field_tag == "sigma_zz":
        field = extract_field(sigma, layout.sigma_blocks["zz"])
    elif field_tag == "sigma_xz":
        field = extract_field(sigma, layout.sigma_blocks["xz"])
    elif field_tag == "v_x":
        field = extract_field(v, layout.v_blocks["x"])
    elif field_tag == "v_z":
        field = extract_field(v, layout.v_blocks["z"])
    else:
        raise ValueError("Unknown field.")

    fig, ax = plt.subplots(figsize=(6, 5))

    VMAX = 0.15
    im = ax.imshow(
        field.T,
        origin="lower",
        extent=[0, Lx, 0, Lz],
        aspect="auto",
        vmin=-VMAX,
        vmax=VMAX,
        cmap="seismic",
    )
    plt.colorbar(im, ax=ax)

    x0 = Lx / 2
    z0 = Lz / 2
    circ_p = Circle((x0, z0), V_P * t, fill=False, linestyle="--", linewidth=2)
    circ_s = Circle((x0, z0), V_S * t, fill=False, linestyle=":", linewidth=2)

    ax.add_patch(circ_p)
    ax.add_patch(circ_s)

    ax.plot(x0, z0, "ko", markersize=4)
    ax.text(x0 + 0.01 * Lx, z0 + V_P * t + 0.01 * Lz, "P", fontsize=12)
    ax.text(x0 + 0.01 * Lx, z0 + V_S * t + 0.01 * Lz, "S", fontsize=12)

    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Lz)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(rf"{field_tag}, $t$={t:.4f}, $V_P t$={V_P*t:.4f}, $V_S t$={V_S*t:.4f}")

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close(fig)


def save_field_frame(writer, field_gif, sigma, v, layout: Layout2D, V_P, V_S, Lx, Lz, n):
    if field_gif == "sigma_xx":
        field = extract_field(sigma, layout.sigma_blocks["xx"])
    elif field_gif == "sigma_zz":
        field = extract_field(sigma, layout.sigma_blocks["zz"])
    elif field_gif == "sigma_xz":
        field = extract_field(sigma, layout.sigma_blocks["xz"])
    elif field_gif == "v_x":
        field = extract_field(v, layout.v_blocks["x"])
    elif field_gif == "v_z":
        field = extract_field(v, layout.v_blocks["z"])
    else:
        raise ValueError("Unknown field.")

    fig, ax = plt.subplots(figsize=(6, 5))
    VMAX = 0.15

    im = ax.imshow(
        field.T,
        origin="lower",
        extent=[0, Lx, 0, Lz],
        aspect="auto",
        vmin=-VMAX,
        vmax=VMAX,
        cmap="seismic",
    )
    plt.colorbar(im, ax=ax)
    plt.title(rf"{field_gif} at step {n}, $V_S={V_S}, V_P={V_P}$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$z$")

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    writer.append_data(img[:, :, :3])

    plt.close(fig)



def PPW(trace, dt, h, V_P, V_S, cutoff):
    F = np.fft.rfft(trace)
    freq = np.fft.rfftfreq(len(trace), d=dt)
    A = np.abs(F)

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



#function that does timestepping
def leapfrog(v, sigma, Lx, Lz, dt, Nt, Nx, Nz, rho, V_P, V_S, n_plot, field_gif, SCL, VCL):
    #step sizes
    dx = Lx / (Nx - 1)
    dz = Lz / (Nz - 1)

    hx_p = dx
    hx_d = dx
    hz_p = dz
    hz_d = dz

    lam = rho * (V_P ** 2 - 2 * V_S ** 2)
    mu = rho * V_S ** 2

    D_sigma, D_v, layout = block_diff_ops(SCL, VCL, Nx, Nz, hx_p, hx_d, hz_p, hz_d)
    M_rho, C = material_matrix(layout, rho, lam, mu)
    M_rho_inv = diags(1.0 / M_rho.diagonal(), 0, format="csc")

    x0, z0 = Lx / 2, Lz / 2
    s = 0.05

    q_v = np.zeros(len(v))
    q_sigma = np.zeros(len(sigma))
    # q_sigma = pressure_source(layout, Lx, Lz, x0, z0, s, amp=1.0)

    #PPW
    trace = np.zeros(Nt)
    field = extract_field(v, layout.v_blocks['x']) #v_x
    nx,nz = field.shape
    ix = int(0.3*nx) 
    iz = int(0.4*nz)
    

    writer = imageio.get_writer(f"{field_gif}_waveP{V_P}S{V_S}.gif", mode="I", fps=20)

    for n in range(Nt):
        t = n * dt

        #1. half-integer stress update 
        sigma = sigma + dt * (C @ (q_sigma + D_v @ v)) #minus sign factored out

        #2. integer velocity update
        v = v + dt * (M_rho_inv @ (q_v + D_sigma @ sigma))

        if n % n_plot == 0:
            save_field_frame(writer, field_gif, sigma, v, layout, V_P, V_S, Lx, Lz, n)

        if n == min(850, Nt - 1):
            save_snapshot(f"{field_gif}_snapshot.png", field_gif, sigma, v, layout, V_P, V_S, Lx, Lz, n, t)

        #PPW
      
        field = extract_field(v, layout.v_blocks['x']) #updated field
        trace[n] = field[iz, iz]


        # if (n == 100) or (n == 150) or (n == 300) or (n == 400) or (n == 500):
        #     v_x = extract_field(v, layout.v_blocks['x'])
        #     plot_cross_section(v_x, Lx, Lz, 'x', iz,  n, title=f'vx slice at n={n}')
       


    writer.close()
    i0 = Nt // 5      # skip early time
    i1 = Nt // 2      # before reflections

    trace_win = trace[i0:i1] #restrict time window 


    PPW(trace_win, dt, dx, V_P, V_S, 0.1)
    return v, sigma, layout







def plot_cross_section(field, Lx, Lz, axis, idx, n, title=''):
    nx, nz = field.shape
    if axis == 'x':
        x = np.linspace(0,Lx,nx)
        plt.plot(x,field[:,idx])
        plt.xlabel('x')
        plt.ylabel('field value')
        plt.title(f'{title} z-index={idx}')
    elif axis == 'x':
        z = np.linspace(0,Lz,nz)
        plt.plot(z,field[idx,:])
        plt.xlabel('z')
        plt.ylabel('field value')
        plt.title(f'{title} x-index={idx}')
    else: 
        raise ValueError('axis must be x or z')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'v_x_cross_section_n{n}.png')
    plt.show()
    
    












def main():

    #STRESS-VELOCITY CLUSTER PAIRS
    SCL = (1, 0, 1)
    VCL = (0, 1, 0)

  
    # SCL = (0, 0, 0)
    # VCL = (1, 1, 1)

    # domain and grid
    Lx, Lz = 3.0, 3.0
    Nx, Nz = 150, 150

    # build layout once to size global state vectors correctly
    layout0 = build_layout_from_clusters(SCL, VCL, Nx, Nz)

    N_sigma = sum(block.size for block in layout0.sigma_blocks.values())
    N_v = sum(block.size for block in layout0.v_blocks.values())

    #material
    rho = 1.0
    V_P = 3.0
    V_S = 1.0

    dx = Lx / (Nx - 1)
    dt =  0.1 * dx / (np.sqrt(2) * V_P)

    T = 1.0
    Nt = int(np.ceil(T / dt))

    #state vectors
    v = np.zeros(N_v)
    sigma = np.zeros(N_sigma)

    #initial condition
    sigma = initial_condition_sigma(sigma, layout0, Lx, Lz, Nx, Nz, A=1.0, s=0.05)

    #run
    n_plot = 10
    field_gif = "v_x"
    v, sigma, layout = leapfrog(v, sigma, Lx, Lz, dt, Nt, Nx, Nz, rho, V_P, V_S, n_plot, field_gif, SCL, VCL)

    #extract fields
    sigma_xx = extract_field(sigma, layout.sigma_blocks["xx"])
    sigma_zz = extract_field(sigma, layout.sigma_blocks["zz"])
    sigma_xz = extract_field(sigma, layout.sigma_blocks["xz"])
    v_x = extract_field(v, layout.v_blocks["x"])
    v_z = extract_field(v, layout.v_blocks["z"])

   
    print(f"Simulating with V_P={V_P}, V_S={V_S}")
    print(f"dt={dt} and Nt={Nt}")
    print("Done.")
    print("sigma_xx shape:", sigma_xx.shape, "grid:", layout.sigma_fields["xx"].grid)
    print("sigma_zz shape:", sigma_zz.shape, "grid:", layout.sigma_fields["zz"].grid)
    print("sigma_xz shape:", sigma_xz.shape, "grid:", layout.sigma_fields["xz"].grid)
    print("v_x shape:", v_x.shape, "grid:", layout.v_fields["x"].grid)
    print("v_z shape:", v_z.shape, "grid:", layout.v_fields["z"].grid)


if __name__ == "__main__":
    main()