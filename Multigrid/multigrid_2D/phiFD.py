import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg

pp = print


class Mesh:
    def __init__(self, N):
        self.Nx = N
        self.Ny = N

        self.x = np.linspace(0.0, 1.0, self.Nx + 1)
        self.y = np.linspace(0.0, 1.0, self.Ny + 1)

        self.hx = self.x[1] - self.x[0]
        self.hy = self.y[1] - self.y[0]
        self.X, self.Y = np.meshgrid(self.x, self.y)


import sympy


def problem_with_solution(case, compute_std_fem=False):
    x_symb, y_symb, z_symb = sympy.symbols("xx yy zz")

    if case == "circle_cos":
        R = 0.3
        phi = lambda x, y: ((x - 0.5) / R) ** 2 + ((y - 0.5) / R) ** 2 - 1.0
        r = sympy.sqrt(
            ((x_symb - 0.5) * (x_symb - 0.5) / (R**2))
            + (((y_symb - 0.5) * (y_symb - 0.5)) / (R**2))
            + 1e-12
        )
        K = sympy.pi / 2.0
        u1 = sympy.cos(K * r)
        f1 = -(
            sympy.diff(sympy.diff(u1, x_symb), x_symb)
            + sympy.diff(sympy.diff(u1, y_symb), y_symb)
        )
    elif case == "ellipsis_cos":
        R_x = 0.41
        R_y = 0.3
        phi = lambda x, y: ((x - 0.5) / R_x) ** 2 + ((y - 0.5) / R_y) ** 2 - 1.0
        r = sympy.sqrt(
            ((x_symb - 0.5) * (x_symb - 0.5) / (R_x**2))
            + (((y_symb - 0.5) * (y_symb - 0.5)) / (R_y**2))
            + 1e-12
        )
        K = sympy.pi / 2.0
        u1 = sympy.cos(K * r)
        f1 = -(
            sympy.diff(sympy.diff(u1, x_symb), x_symb)
            + sympy.diff(sympy.diff(u1, y_symb), y_symb)
        )
    elif case == "ellipsis_cos_exp":
        R_x = 0.25
        R_y = np.sqrt(2.0) / 4.0
        phi = lambda x, y: ((x - 0.5) / R_x) ** 2 + ((y - 0.5) / R_y) ** 2 - 1.0
        r = sympy.sqrt(
            ((x_symb - 0.5) / R_x) ** 2 + ((y_symb - 0.5) / R_y) ** 2 + 1e-12
        )
        K = sympy.pi / 2.0
        u1 = sympy.cos(K * r) * sympy.exp(x_symb) * sympy.sin(y_symb * x_symb)
        f1 = -(
            sympy.diff(sympy.diff(u1, x_symb), x_symb)
            + sympy.diff(sympy.diff(u1, y_symb), y_symb)
        )

    else:
        raise Exception(f"unknown case:{case}")

    uref_fn = sympy.lambdify([x_symb, y_symb], u1)
    f = sympy.lambdify([x_symb, y_symb], f1)

    uref_fn_numpy = lambda x, y: uref_fn(x, y).ravel()
    if compute_std_fem:
        return u1, f1
    else:
        return phi, f, uref_fn_numpy


def build_matrices(mesh: Mesh, phi):
    sigma = 0.01
    gamma = 10.0
    if callable(phi):
        phiji = phi(mesh.X, mesh.Y)
    else:
        assert type(phi) == np.ndarray
        phiji = phi
    ind = (phiji < 0) + 0
    mask = sp.diags(diagonals=ind.ravel()).tocsr()
    indOut = 1 - ind
    Ndof = (mesh.Nx + 1) * (mesh.Ny + 1)

    # laplacian matrix
    D2x = (1.0 / mesh.hx / mesh.hx) * sp.diags(
        diagonals=[-1, 2, -1], offsets=[-1, 0, 1], shape=(mesh.Nx + 1, mesh.Nx + 1)
    )
    D2y = (1.0 / mesh.hy / mesh.hy) * sp.diags(
        diagonals=[-1, 2, -1], offsets=[-1, 0, 1], shape=(mesh.Ny + 1, mesh.Ny + 1)
    )
    D2x_2d = sp.kron(sp.eye(mesh.Ny + 1), D2x).tocsr()
    D2y_2d = sp.kron(D2y, sp.eye(mesh.Nx + 1)).tocsr()

    A = mask @ (D2x_2d + D2y_2d)
    # boundary conditions
    row = []
    col = []
    coef = []  # for the matrix implementing BC

    def rav(i, j):
        return np.ravel_multi_index([j, i], (mesh.Ny + 1, mesh.Nx + 1))

    def AddMat(eq, i, j, a):
        row.append(eq)
        col.append(rav(i, j))
        coef.append(a)

    # active sites for the ghost penalty
    actGx = np.zeros((mesh.Ny + 1, mesh.Nx + 1))
    actGy = np.zeros((mesh.Ny + 1, mesh.Nx + 1))

    indx = ind[:, 1 : mesh.Nx + 1] - ind[:, 0 : mesh.Nx]
    J, I = np.where((indx == 1) | (indx == -1))
    for k in range(np.shape(I)[0]):
        i, j = I[k], J[k]
        if indx[j, i] == 1:
            indOut[j, i] = 0
            actGx[j, i + 1] = 1
        else:
            indOut[j, i + 1] = 0
            actGx[j, i] = 1
        phiS = np.square(phiji[j, i]) + np.square(phiji[j, i + 1])
        AddMat(rav(i, j), i, j, phiji[j, i + 1] * phiji[j, i + 1] / phiS)
        AddMat(rav(i, j), i + 1, j, -phiji[j, i] * phiji[j, i + 1] / phiS)
        AddMat(rav(i + 1, j), i, j, -phiji[j, i] * phiji[j, i + 1] / phiS)
        AddMat(rav(i + 1, j), i + 1, j, phiji[j, i] * phiji[j, i] / phiS)

    indy = ind[1 : mesh.Ny + 1, :] - ind[0 : mesh.Ny, :]
    J, I = np.where((indy == 1) | (indy == -1))
    for k in range(np.shape(I)[0]):
        i, j = I[k], J[k]
        if indy[j, i] == 1:
            indOut[j, i] = 0
            actGy[j + 1, i] = 1
        else:
            indOut[j + 1, i] = 0
            actGy[j, i] = 1
        phiS = np.square(phiji[j, i]) + np.square(phiji[j + 1, i])
        AddMat(rav(i, j), i, j, phiji[j + 1, i] * phiji[j + 1, i] / phiS)
        AddMat(rav(i, j), i, j + 1, -phiji[j, i] * phiji[j + 1, i] / phiS)
        AddMat(rav(i, j + 1), i, j, -phiji[j, i] * phiji[j + 1, i] / phiS)
        AddMat(rav(i, j + 1), i, j + 1, phiji[j, i] * phiji[j, i] / phiS)

    npcoef = (gamma / mesh.hx / mesh.hy) * np.array(coef)
    B = sp.coo_matrix((npcoef, (row, col)), shape=(Ndof, Ndof))
    # ghost penalty
    maskGx = sp.diags(diagonals=actGx.ravel())
    maskGy = sp.diags(diagonals=actGy.ravel())

    C = sigma * (
        mesh.hx**2 * (D2x_2d.T @ maskGx @ D2x_2d)
        + mesh.hy**2 * (D2y_2d.T @ maskGy @ D2y_2d)
    )
    # penalization outside
    D = sp.diags(diagonals=indOut.ravel())
    # linear system
    A = (A + B + C + (mesh.hx * mesh.hy) ** (-1) * D).tocsr()  #
    return ind.ravel(), A


def force(mesh, ind, f):
    b = f(mesh.X, mesh.Y).ravel()
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

    u = linalg.spsolve(A, b)
    print(errors_L2_Loo_fn(ind, u, uref_fn(mesh.X, mesh.Y), mesh))
