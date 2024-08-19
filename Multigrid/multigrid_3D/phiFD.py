import numpy as np
import scipy.sparse as sp

pp = print
import pylab as plt


class Mesh:
    def __init__(self, N):
        self.Nx = N
        self.Ny = N
        self.Nz = N

        self.x = np.linspace(0.0, 1.0, self.Nx + 1)
        self.y = np.linspace(0.0, 1.0, self.Ny + 1)
        self.z = np.linspace(0.0, 1.0, self.Nz + 1)

        self.hx = self.x[1] - self.x[0]
        self.hy = self.y[1] - self.y[0]
        self.hz = self.z[1] - self.z[0]
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z)


import sympy


def problem_with_solution(case, compute_std_fem=False):
    x_symb, y_symb, z_symb = sympy.symbols("xx yy zz")

    if case == "sphere_cos":
        R = 0.3
        phi = (
            lambda x, y, z: ((x - 0.5) / R) ** 2
            + ((y - 0.5) / R) ** 2
            + ((z - 0.5) / R) ** 2
            - 1.0
        )
        r = sympy.sqrt(
            ((x_symb - 0.5) * (x_symb - 0.5) / (R**2))
            + (((y_symb - 0.5) * (y_symb - 0.5)) / (R**2))
            + (((z_symb - 0.5) * (z_symb - 0.5)) / (R**2))
            + 1e-12
        )
        K = sympy.pi / 2.0
        u1 = sympy.cos(K * r)
        f1 = -(
            sympy.diff(sympy.diff(u1, x_symb), x_symb)
            + sympy.diff(sympy.diff(u1, y_symb), y_symb)
            + sympy.diff(sympy.diff(u1, z_symb), z_symb)
        )

    else:
        raise Exception(f"unknown case:{case}")

    uref_fn = sympy.lambdify([x_symb, y_symb, z_symb], u1)
    f = sympy.lambdify([x_symb, y_symb, z_symb], f1)

    uref_fn_numpy = lambda x, y, z: uref_fn(x, y, z).ravel()
    if compute_std_fem:
        return u1, f1
    else:
        return phi, f, uref_fn_numpy


import time


def build_matrices(mesh: Mesh, phi):
    sigma = 0.01
    gamma = 10.0
    if callable(phi):
        phijik = phi(mesh.X, mesh.Y, mesh.Z)
    else:
        assert type(phi) == np.ndarray
        phijik = phi

    ind = (phijik < 0) + 0
    mask = sp.diags(diagonals=ind.ravel()).tocsr()
    indOut = 1 - ind
    Ndof = (mesh.Nx + 1) * (mesh.Ny + 1) * (mesh.Nz + 1)

    # laplacian matrix
    D2x = (1.0 / mesh.hx / mesh.hx) * sp.diags(
        diagonals=[-1, 2, -1], offsets=[-1, 0, 1], shape=(mesh.Nx + 1, mesh.Nx + 1)
    )
    D2y = (1.0 / mesh.hy / mesh.hy) * sp.diags(
        diagonals=[-1, 2, -1], offsets=[-1, 0, 1], shape=(mesh.Ny + 1, mesh.Ny + 1)
    )
    D2z = (1.0 / mesh.hz / mesh.hz) * sp.diags(
        diagonals=[-1, 2, -1], offsets=[-1, 0, 1], shape=(mesh.Nz + 1, mesh.Nz + 1)
    )

    # 3D matrix operators from 1D operators,: using kronecker product
    D2x_2d = sp.kron(sp.kron(sp.eye(mesh.Ny + 1), D2x), sp.eye(mesh.Nz + 1)).tocsr()
    D2y_2d = sp.kron(sp.kron(D2y, sp.eye(mesh.Nx + 1)), sp.eye(mesh.Nz + 1)).tocsr()
    D2z_2d = sp.kron(sp.kron(sp.eye(mesh.Ny + 1), sp.eye(mesh.Nx + 1)), D2z).tocsr()

    A = mask @ (D2x_2d + D2y_2d + D2z_2d)
    # boundary conditions
    row = []
    col = []
    coef = []

    def rav(i, j, k):
        return np.ravel_multi_index([j, i, k], (mesh.Ny + 1, mesh.Nx + 1, mesh.Nz + 1))

    def AddMat(eq, i, j, k, a):
        row.append(eq)
        col.append(rav(i, j, k))
        coef.append(a)

    # active sites for the ghost penalty
    actGx = np.zeros((mesh.Ny + 1, mesh.Nx + 1, mesh.Nz + 1))
    actGy = np.zeros((mesh.Ny + 1, mesh.Nx + 1, mesh.Nz + 1))
    actGz = np.zeros((mesh.Ny + 1, mesh.Nx + 1, mesh.Nz + 1))

    indx = ind[:, 1 : mesh.Nx + 1, :] - ind[:, 0 : mesh.Nx, :]
    J, I, K = np.where((indx == 1) | (indx == -1))
    for j, i, k_ in zip(J, I, K):
        if indx[j, i, k_] == 1:
            indOut[j, i, k_] = 0
            actGx[j, i + 1, k_] = 1
        else:
            indOut[j, i + 1, k_] = 0
            actGx[j, i, k_] = 1

        # i, j, k_ = I[k], J[k], K[k]
        phiS = np.square(phijik[j, i, k_]) + np.square(phijik[j, i + 1, k_])
        phii = phijik[j, i, k_]
        phiip1 = phijik[j, i + 1, k_]
        AddMat(rav(i, j, k_), i, j, k_, phiip1 * phiip1 / phiS)
        AddMat(rav(i, j, k_), i + 1, j, k_, -phii * phiip1 / phiS)
        AddMat(rav(i + 1, j, k_), i, j, k_, -phii * phiip1 / phiS)
        AddMat(rav(i + 1, j, k_), i + 1, j, k_, phii * phii / phiS)

    indy = ind[1 : mesh.Ny + 1, :, :] - ind[0 : mesh.Ny, :, :]
    J, I, K = np.where((indy == 1) | (indy == -1))
    for j, i, k_ in zip(J, I, K):
        if indy[j, i, k_] == 1:
            indOut[j, i, k_] = 0
            actGy[j + 1, i, k_] = 1
        else:
            indOut[j + 1, i, k_] = 0
            actGy[j, i, k_] = 1

        phiS = np.square(phijik[j, i, k_]) + np.square(phijik[j + 1, i, k_])
        phij = phijik[j, i, k_]
        phijp1 = phijik[j + 1, i, k_]
        AddMat(rav(i, j, k_), i, j, k_, phijp1 * phijp1 / phiS)
        AddMat(rav(i, j, k_), i, j + 1, k_, -phij * phijp1 / phiS)
        AddMat(rav(i, j + 1, k_), i, j, k_, -phij * phijp1 / phiS)
        AddMat(rav(i, j + 1, k_), i, j + 1, k_, phij * phij / phiS)

    indz = ind[:, :, 1 : mesh.Nz + 1] - ind[:, :, 0 : mesh.Nz]
    J, I, K = np.where((indz == 1) | (indz == -1))
    for j, i, k_ in zip(J, I, K):
        if indz[j, i, k_] == 1:
            indOut[j, i, k_] = 0
            actGz[j, i, k_ + 1] = 1
        else:
            indOut[j, i, k_ + 1] = 0
            actGz[j, i, k_] = 1

        phiS = np.square(phijik[j, i, k_]) + np.square(phijik[j, i, k_ + 1])
        phik = phijik[j, i, k_]
        phikp1 = phijik[j, i, k_ + 1]
        AddMat(rav(i, j, k_), i, j, k_, phikp1 * phikp1 / phiS)
        AddMat(rav(i, j, k_), i, j, k_ + 1, -phik * phikp1 / phiS)
        AddMat(rav(i, j, k_ + 1), i, j, k_, -phik * phikp1 / phiS)
        AddMat(rav(i, j, k_ + 1), i, j, k_ + 1, phik * phik / phiS)
    npcoef = (gamma / mesh.hx / mesh.hy) * np.array(coef)
    B = sp.coo_array((npcoef, (row, col)), shape=(Ndof, Ndof))
    # ghost penalty
    maskGx = sp.diags(diagonals=actGx.ravel())
    maskGy = sp.diags(diagonals=actGy.ravel())
    maskGz = sp.diags(diagonals=actGz.ravel())

    C = sigma * (
        mesh.hx**2 * (D2x_2d.T @ maskGx @ D2x_2d)
        + mesh.hy**2 * (D2y_2d.T @ maskGy @ D2y_2d)
        + mesh.hz**2 * (D2z_2d.T @ maskGz @ D2z_2d)
    )
    # penalization outside
    D = sp.diags(diagonals=indOut.ravel())
    # linear system
    A = (A + B + C + (mesh.hx * mesh.hy) ** (-1) * D).tocsr()
    return ind.ravel(), A


def force(mesh, ind, f):
    b = f(mesh.X, mesh.Y, mesh.Z).ravel()
    b *= ind
    return b


def errors_L2_Loo_fn(ind, u, uref, mesh):

    e = ind * (u - uref)
    eL2 = np.sqrt(np.sum(e * e)) / np.sqrt(np.sum(uref * uref * ind))
    emax = np.max(np.abs(e * (ind))) / np.max(np.abs(uref * (ind)))

    return {"L2": eL2, "Loo": emax}


if __name__ == "__main__":
    phi, f, uref_fn = problem_with_solution("circle_cos")
    mesh = Mesh(48)
    ind, A = build_matrices(mesh=mesh, phi=phi)
    b = force(mesh, ind, f)
    print(f"{mesh.Nx=}")
    print(f"{(mesh.Nx+1)**3=}")
    print(f"{A.shape=}")
    print(f"{ind.shape=}")
    # u = sp.linalg.spsolve(A, b)
    # print(errors_L2_Loo_fn(ind, u, uref_fn(mesh.X, mesh.Y, mesh.Z), mesh))
