import numpy as np
import dolfin as df
import sympy
import matplotlib.pyplot as plt
from time import time
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import pylab as py
import seaborn as sns

sns.set_theme("paper")

df.parameters["ghost_mode"] = "shared_facet"
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True
df.parameters["allow_extrapolation"] = True
df.parameters["form_compiler"]["representation"] = "uflacs"

# radius of the domains
R_x = 0.3 + 1e-10
R_y = 0.3 + 1e-10

# Compute the conditioning number
conditioning = True

# Number of iterations
init_Iter = 1
Iter = 4 if conditioning else 6

# Polynome Pk
polV = 1
polPhi = polV + 1
# parameters["form_compiler"]["quadrature_degree"]=2*(polV+polPhi)

# Ghost penalty
ghost = True

# plot the solution
Plot = False


def Omega(x, y):
    return ((x - 0.5) / R_x) ** 2 + ((y - 0.5) / R_y) ** 2 <= 1.0 + df.DOLFIN_EPS


# Function used to write in the outputs files
def output_latex(f, A, B):
    for i in range(len(A)):
        f.write("(")
        f.write(str(A[i]))
        f.write(",")
        f.write(str(B[i]))
        f.write(")\n")
    # f.write("\n")


# Computation of the Exact solution and exact source term
x_symb, y_symb = sympy.symbols("xx yy")
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

nabla_u_x = sympy.diff(u1, x_symb)
nabla_u_y = sympy.diff(u1, y_symb)
nabla_u_x_np = sympy.lambdify([x_symb, y_symb], nabla_u_x)
nabla_u_y_np = sympy.lambdify([x_symb, y_symb], nabla_u_y)

###########################################
### beginning phi fem ###########
###########################################

# parameter of the ghost penalty
sigma = 1.0

# Initialistion of the output
size_mesh = np.zeros(Iter)
NN = []
for i in range(init_Iter - 1, Iter):
    print("#########################")
    print("## Iteration phifem ", i + 1, "##")
    print("#########################")

    # Construction of the mesh
    N = int(10 * 2 ** ((i)))
    NN.append(N)
    print("N=", N)
    mesh_macro = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(1.0, 1.0), N, N)
    size_mesh[i] = mesh_macro.hmax()

###########################################
####### beginning finite difference #######
#######        first  scheme        #######
###########################################
Gamma = [0.001, 0.01, 0.1, 1.0, 10.0, 20.0]
Sigma = [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 20.0]
# Initialistion of the output
size_mesh_diff_vec_1 = []
error_L2_diff_vec_1 = []
error_Linf_diff_vec_1 = []
error_H1_diff_vec_1 = []
error_H1int_diff_vec_1 = []
cond_diff_vec_1 = []

for ii in range(len(NN)):
    print("######################")
    print("## Iteration phiFD 1 ", ii + 1, "##")
    print("######################")
    # Construction of the mesh
    N = NN[ii]
    print("N=", N)
    Nx = N
    Ny = N
    x = np.linspace(0.0, 1.0, Nx + 1)
    y = np.linspace(0.0, 1.0, Ny + 1)

    ue = sympy.lambdify([x_symb, y_symb], u1)
    f = sympy.lambdify([x_symb, y_symb], f1)
    phi = lambda x, y: -1.0 + ((x - 0.5) / R_x) ** 2 + ((y - 0.5) / R_y) ** 2

    Ndof = (Nx + 1) * (Ny + 1)
    hx = x[1] - x[0]
    hy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)  # 2D meshgrid
    phiij = phi(X, Y)
    ind = (phiij < 0) + 0
    mask = sp.diags(diagonals=ind.ravel())
    indOut = 1 - ind

    uref = ue(X, Y)
    error_L2_diff_vec_change_gamma = []
    error_Linf_diff_vec_change_gamma = []
    error_H1_diff_vec_change_gamma = []
    error_H1int_diff_vec_change_gamma = []
    cond_diff_vec_change_gamma = []
    for gamma in Gamma:
        error_L2_diff_vec_change_sigma = []
        error_Linf_diff_vec_change_sigma = []
        error_H1_diff_vec_change_sigma = []
        error_H1int_diff_vec_change_sigma = []
        cond_diff_vec_change_sigma = []
        for sigma in Sigma:
            print(f"{sigma=} {gamma=}")
            D2x = (1 / hx / hx) * sp.diags(
                diagonals=[-1, 2, -1], offsets=[-1, 0, 1], shape=(Nx + 1, Nx + 1)
            )
            D2y = (1 / hy / hy) * sp.diags(
                diagonals=[-1, 2, -1], offsets=[-1, 0, 1], shape=(Ny + 1, Ny + 1)
            )
            D2x_2d = sp.kron(sp.eye(Ny + 1), D2x)
            D2y_2d = sp.kron(D2y, sp.eye(Nx + 1))
            A = mask @ (D2x_2d + D2y_2d)
            row = []
            col = []
            coef = []

            def rav(i, j):
                return np.ravel_multi_index([j, i], (Ny + 1, Nx + 1))

            def AddMat(eq, i, j, a):
                row.append(eq)
                col.append(i + (Nx + 1) * j)
                coef.append(a)

            actGx = np.zeros((Ny + 1, Nx + 1))
            actGy = np.zeros((Ny + 1, Nx + 1))

            indx = ind[:, 1 : Nx + 1] - ind[:, 0:Nx]
            J, I = np.where((indx == 1) | (indx == -1))
            for k in range(np.shape(I)[0]):
                if indx[J[k], I[k]] == 1:
                    indOut[J[k], I[k]] = 0
                    actGx[J[k], I[k] + 1] = 1
                else:
                    indOut[J[k], I[k] + 1] = 0
                    actGx[J[k], I[k]] = 1
                phiS = np.square(phiij[J[k], I[k]]) + np.square(phiij[J[k], I[k] + 1])
                AddMat(
                    I[k] + (Nx + 1) * J[k],
                    I[k],
                    J[k],
                    phiij[J[k], I[k] + 1] * phiij[J[k], I[k] + 1] / phiS,
                )
                AddMat(
                    I[k] + (Nx + 1) * J[k],
                    I[k] + 1,
                    J[k],
                    -phiij[J[k], I[k]] * phiij[J[k], I[k] + 1] / phiS,
                )
                AddMat(
                    I[k] + 1 + (Nx + 1) * J[k],
                    I[k] + 1,
                    J[k],
                    phiij[J[k], I[k]] * phiij[J[k], I[k]] / phiS,
                )
                AddMat(
                    I[k] + 1 + (Nx + 1) * J[k],
                    I[k],
                    J[k],
                    -phiij[J[k], I[k]] * phiij[J[k], I[k] + 1] / phiS,
                )

            indy = ind[1 : Ny + 1, :] - ind[0:Ny, :]
            J, I = np.where((indy == 1) | (indy == -1))
            for k in range(np.shape(I)[0]):
                if indy[J[k], I[k]] == 1:
                    indOut[J[k], I[k]] = 0
                    actGy[J[k] + 1, I[k]] = 1
                else:
                    indOut[J[k] + 1, I[k]] = 0
                    actGy[J[k], I[k]] = 1
                phiS = np.square(phiij[J[k], I[k]]) + np.square(phiij[J[k] + 1, I[k]])
                AddMat(
                    I[k] + (Nx + 1) * J[k],
                    I[k],
                    J[k],
                    phiij[J[k] + 1, I[k]] * phiij[J[k] + 1, I[k]] / phiS,
                )
                AddMat(
                    I[k] + (Nx + 1) * J[k],
                    I[k],
                    J[k] + 1,
                    -phiij[J[k], I[k]] * phiij[J[k] + 1, I[k]] / phiS,
                )
                AddMat(
                    I[k] + (Nx + 1) * (J[k] + 1),
                    I[k],
                    J[k] + 1,
                    phiij[J[k], I[k]] * phiij[J[k], I[k]] / phiS,
                )
                AddMat(
                    I[k] + (Nx + 1) * (J[k] + 1),
                    I[k],
                    J[k],
                    -phiij[J[k], I[k]] * phiij[J[k] + 1, I[k]] / phiS,
                )

            maskGx = sp.diags(diagonals=actGx.ravel())
            maskGy = sp.diags(diagonals=actGy.ravel())
            b = f(X, Y)
            b = (ind * b).ravel()
            D = sp.diags(diagonals=indOut.ravel())
            npcoef = (gamma / hx / hy) * np.array(coef)
            B = sp.coo_array((npcoef, (row, col)), shape=(Ndof, Ndof))
            C = (
                sigma
                * hx
                * hy
                * (D2x_2d.T @ maskGx @ D2x_2d + D2y_2d.T @ maskGy @ D2y_2d)
            )

            A = (A + B + C + D).tocsr()
            u = spsolve(A, b).reshape(Ny + 1, Nx + 1)
            e = u - uref

            eL2 = np.sqrt(np.sum(e * e * (1 - indOut)) * hx * hy) / np.sqrt(
                np.sum(uref * uref * (1 - indOut)) * hx * hy
            )
            emax = np.max(np.abs(e * (1 - indOut))) / np.max(
                np.abs(uref * (1 - indOut))
            )

            ex = (e[:, 1 : Nx + 1] - e[:, 0:Nx]) / hx
            urefx = (uref[:, 1 : Nx + 1] - uref[:, 0:Nx]) / hx
            intdx = (ind[:, 1 : Nx + 1] + ind[:, 0:Nx] == 2) + 0
            fulldx = (ind[:, 1 : Nx + 1] + ind[:, 0:Nx] > 0) + 0

            ey = (e[1 : Ny + 1, :] - e[0:Ny, :]) / hy
            urefy = (uref[1 : Ny + 1, :] - uref[0:Ny, :]) / hy
            intdy = (ind[1 : Ny + 1, :] + ind[0:Ny, :] == 2) + 0
            fulldy = (ind[1 : Ny + 1, :] + ind[0:Ny, :] > 0) + 0

            eH1 = np.sqrt(
                (np.sum(ex * ex * fulldx) + np.sum(ey * ey * fulldy)) * hx * hy
            ) / np.sqrt(
                (np.sum(urefx * urefx * fulldx) + np.sum(urefy * urefy * fulldy))
                * hx
                * hy
            )
            eH1int = np.sqrt(
                (np.sum(ex * ex * intdx) + np.sum(ey * ey * intdy)) * hx * hy
            ) / np.sqrt(
                (np.sum(urefx * urefx * intdx) + np.sum(urefy * urefy * intdy))
                * hx
                * hy
            )

            if conditioning == True:
                cond = np.linalg.cond(A.todense())
                cond_diff_vec_change_sigma.append(cond)
            error_L2_diff_vec_change_sigma.append(eL2)
            error_Linf_diff_vec_change_sigma.append(emax)
            error_H1_diff_vec_change_sigma.append(eH1)
            error_H1int_diff_vec_change_sigma.append(eH1int)
        error_L2_diff_vec_change_gamma.append(error_L2_diff_vec_change_sigma)
        error_Linf_diff_vec_change_gamma.append(error_Linf_diff_vec_change_sigma)
        error_H1_diff_vec_change_gamma.append(error_H1_diff_vec_change_sigma)
        error_H1int_diff_vec_change_gamma.append(error_H1int_diff_vec_change_sigma)
        if conditioning == True:
            cond_diff_vec_change_gamma.append(cond_diff_vec_change_sigma)
    error_L2_diff_vec_1.append(error_L2_diff_vec_change_gamma)
    error_Linf_diff_vec_1.append(error_Linf_diff_vec_change_gamma)
    error_H1_diff_vec_1.append(error_H1_diff_vec_change_gamma)
    error_H1int_diff_vec_1.append(error_H1int_diff_vec_change_gamma)
    if conditioning == True:
        cond_diff_vec_1.append(cond_diff_vec_change_gamma)
error_L2_diff_array_1 = np.array(error_L2_diff_vec_1)
error_Linf_diff_array_1 = np.array(error_Linf_diff_vec_1)
error_H1_diff_array_1 = np.array(error_H1_diff_vec_1)
error_H1int_diff_array_1 = np.array(error_H1int_diff_vec_1)
if conditioning == True:
    cond_diff_array_1 = np.array(cond_diff_vec_1)
size_mesh_array_1 = np.zeros_like(error_L2_diff_array_1)
size_mesh_array_1[:, :, :] = size_mesh[:, None, None]


if conditioning == True:
    full_results_array_1 = np.array(
        [
            size_mesh_array_1,
            error_L2_diff_array_1,
            error_Linf_diff_array_1,
            error_H1_diff_array_1,
            error_H1int_diff_array_1,
            cond_diff_array_1,
        ]
    )
    np.save("full_res_with_cond_phiFD1.npy", full_results_array_1)
else:

    full_results_array_1 = np.array(
        [
            size_mesh_array_1,
            error_L2_diff_array_1,
            error_Linf_diff_array_1,
            error_H1_diff_array_1,
            error_H1int_diff_array_1,
        ]
    )

    np.save("full_res_without_cond_phiFD1.npy", full_results_array_1)
print(f"{full_results_array_1.shape=}")


interp_errors_2 = []
size_mesh_diff_vec_2 = []
error_L2_diff_vec_2 = []
error_Linf_diff_vec_2 = []
error_H1_diff_vec_2 = []
error_H1int_diff_vec_2 = []
cond_diff_vec_2 = []

for iii in range(len(NN)):
    print("######################")
    print("## Iteration phiFD 2 ", iii + 1, "##")
    print("######################")
    # Construction of the mesh
    N = NN[iii]
    print("N=", N)
    Nx = N
    Ny = N
    x = np.linspace(0.0, 1.0, Nx + 1)
    y = np.linspace(0.0, 1.0, Ny + 1)

    ue = sympy.lambdify([x_symb, y_symb], u1)
    f = sympy.lambdify([x_symb, y_symb], f1)
    phi = lambda x, y: -1.0 + ((x - 0.5) / R_x) ** 2 + ((y - 0.5) / R_y) ** 2

    Ndof = (Nx + 1) * (Ny + 1)
    hx = x[1] - x[0]
    hy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)  # 2D meshgrid
    phiij = phi(X, Y)
    ind = (phiij < 0) + 0
    mask = sp.diags(diagonals=ind.ravel())
    indOut = 1 - ind

    uref = ue(X, Y)
    error_L2_diff_vec_change_gamma = []
    error_Linf_diff_vec_change_gamma = []
    error_H1_diff_vec_change_gamma = []
    error_H1int_diff_vec_change_gamma = []
    cond_diff_vec_change_gamma = []
    for gamma in Gamma:
        error_L2_diff_vec_change_sigma = []
        error_Linf_diff_vec_change_sigma = []
        error_H1_diff_vec_change_sigma = []
        error_H1int_diff_vec_change_sigma = []
        cond_diff_vec_change_sigma = []
        for sigma in Sigma:
            print(f"{sigma=} {gamma=}")

            D2x = (1 / hx / hx) * sp.diags(
                diagonals=[-1, 2, -1], offsets=[-1, 0, 1], shape=(Nx + 1, Nx + 1)
            )
            D2y = (1 / hy / hy) * sp.diags(
                diagonals=[-1, 2, -1], offsets=[-1, 0, 1], shape=(Ny + 1, Ny + 1)
            )

            # 2D matrix operators from 1D operators using kronecker product
            D2x_2d = sp.kron(sp.eye(Ny + 1), D2x)
            D2y_2d = sp.kron(D2y, sp.eye(Nx + 1))

            A = mask @ (D2x_2d + D2y_2d)
            # boundary conditions
            row = []
            col = []
            coef = []  # for the matrix implementing BC
            fb = np.zeros((Ny + 1, Nx + 1))  # for the RHS corresponding to BC

            def AddMat(eq, i, j, a):
                row.append(eq)
                col.append(i + (Nx + 1) * j)
                coef.append(a)

            # active sites for the ghost penalty
            actGx = np.zeros((Ny + 1, Nx + 1))
            actGy = np.zeros((Ny + 1, Nx + 1))

            indx = ind[:, 1 : Nx + 1] - ind[:, 0:Nx]
            J, I = np.where((indx == 1) | (indx == -1))
            for k in range(np.shape(I)[0]):
                if indx[J[k], I[k]] == 1:
                    i, j = I[k] + 1, J[k]
                    indOut[J[k], I[k]] = 0
                    actGx[J[k], I[k] + 1] = 1
                else:
                    i, j = I[k], J[k]
                    indOut[J[k], I[k] + 1] = 0
                    actGx[J[k], I[k]] = 1
                ai = 2 * phiij[j, i + 1] * phiij[j, i - 1]
                aim1 = -phiij[j, i] * phiij[j, i + 1]
                aip1 = -phiij[j, i] * phiij[j, i - 1]
                phiS = ai**2 + aim1**2 + aip1**2
                AddMat(i - 1 + (Nx + 1) * j, i - 1, j, aim1 * aim1 / phiS)
                AddMat(i - 1 + (Nx + 1) * j, i, j, aim1 * ai / phiS)
                AddMat(i - 1 + (Nx + 1) * j, i + 1, j, aim1 * aip1 / phiS)
                AddMat(i + (Nx + 1) * j, i - 1, j, ai * aim1 / phiS)
                AddMat(i + (Nx + 1) * j, i, j, ai * ai / phiS)
                AddMat(i + (Nx + 1) * j, i + 1, j, ai * aip1 / phiS)
                AddMat(i + 1 + (Nx + 1) * j, i - 1, j, aip1 * aim1 / phiS)
                AddMat(i + 1 + (Nx + 1) * j, i, j, aip1 * ai / phiS)
                AddMat(i + 1 + (Nx + 1) * j, i + 1, j, aip1 * aip1 / phiS)

            indy = ind[1 : Ny + 1, :] - ind[0:Ny, :]
            J, I = np.where((indy == 1) | (indy == -1))
            for k in range(np.shape(I)[0]):
                if indy[J[k], I[k]] == 1:
                    i, j = I[k], J[k] + 1
                    indOut[J[k], I[k]] = 0
                    actGy[J[k] + 1, I[k]] = 1
                else:
                    i, j = I[k], J[k]
                    indOut[J[k] + 1, I[k]] = 0
                    actGy[J[k], I[k]] = 1
                ai = 2 * phiij[j + 1, i] * phiij[j - 1, i]
                aim1 = -phiij[j, i] * phiij[j + 1, i]
                aip1 = -phiij[j, i] * phiij[j - 1, i]
                phiS = ai**2 + aim1**2 + aip1**2
                AddMat(i + (Nx + 1) * (j - 1), i, j - 1, aim1 * aim1 / phiS)
                AddMat(i + (Nx + 1) * (j - 1), i, j, aim1 * ai / phiS)
                AddMat(i + (Nx + 1) * (j - 1), i, j + 1, aim1 * aip1 / phiS)
                AddMat(i + (Nx + 1) * j, i, j - 1, ai * aim1 / phiS)
                AddMat(i + (Nx + 1) * j, i, j, ai * ai / phiS)
                AddMat(i + (Nx + 1) * j, i, j + 1, ai * aip1 / phiS)
                AddMat(i + (Nx + 1) * (j + 1), i, j - 1, aip1 * aim1 / phiS)
                AddMat(i + (Nx + 1) * (j + 1), i, j, aip1 * ai / phiS)
                AddMat(i + (Nx + 1) * (j + 1), i, j + 1, aip1 * aip1 / phiS)

            # penalistion of the boundary condition
            npcoef = (gamma / hx / hy) * np.array(coef)
            B = sp.coo_array((npcoef, (row, col)), shape=(Ndof, Ndof))
            row = []
            col = []
            coef = []  # for the matrix implementing BC

            def AddMat(eq, i, j, a):
                row.append(eq)
                col.append(i + (Nx + 1) * j)
                coef.append(a)

            # active sites for the ghost penalty
            actGx = np.zeros((Ny + 1, Nx + 1))
            actGy = np.zeros((Ny + 1, Nx + 1))

            indx = ind[:, 1 : Nx + 1] - ind[:, 0:Nx]
            J, I = np.where((indx == 1) | (indx == -1))
            for k in range(np.shape(I)[0]):
                if indx[J[k], I[k]] == 1:
                    i, j = I[k] + 1, J[k]
                    i_ = i
                else:
                    i, j = I[k], J[k]
                    i_ = i - 1
                if (
                    ind[j, i_ - 1] == 0
                    and ind[j, i_] == 1
                    and ind[j, i_ + 1] == 1
                    and ind[j, i_ + 2] == 1
                ) or (
                    ind[j, i_ - 1] == 1
                    and ind[j, i_] == 1
                    and ind[j, i_ + 1] == 1
                    and ind[j, i_ + 2] == 0
                ):
                    AddMat(i_ - 1 + (Nx + 1) * j, i_ - 1, j, sigma)
                    AddMat(i_ - 1 + (Nx + 1) * j, i_, j, -3 * sigma)
                    AddMat(i_ - 1 + (Nx + 1) * j, i_ + 1, j, 3 * sigma)
                    AddMat(i_ - 1 + (Nx + 1) * j, i_ + 2, j, -sigma)
                    AddMat(i_ + (Nx + 1) * j, i_ - 1, j, -3 * sigma)
                    AddMat(i_ + (Nx + 1) * j, i_, j, 9 * sigma)
                    AddMat(i_ + (Nx + 1) * j, i_ + 1, j, -9 * sigma)
                    AddMat(i_ + (Nx + 1) * j, i_ + 2, j, 3 * sigma)
                    AddMat(i_ + 1 + (Nx + 1) * j, i_ - 1, j, 3 * sigma)
                    AddMat(i_ + 1 + (Nx + 1) * j, i_, j, -9 * sigma)
                    AddMat(i_ + 1 + (Nx + 1) * j, i_ + 1, j, 9 * sigma)
                    AddMat(i_ + 1 + (Nx + 1) * j, i_ + 2, j, -3 * sigma)
                    AddMat(i_ + 2 + (Nx + 1) * j, i_ - 1, j, -sigma)
                    AddMat(i_ + 2 + (Nx + 1) * j, i_, j, 3 * sigma)
                    AddMat(i_ + 2 + (Nx + 1) * j, i_ + 1, j, -3 * sigma)
                    AddMat(i_ + 2 + (Nx + 1) * j, i_ + 2, j, sigma)

            indy = ind[1 : Ny + 1, :] - ind[0:Ny, :]
            J, I = np.where((indy == 1) | (indy == -1))
            for k in range(np.shape(I)[0]):
                if indy[J[k], I[k]] == 1:
                    i, j = I[k], J[k] + 1
                    j_ = j
                else:
                    i, j = I[k], J[k]
                    j_ = j - 1
                if (
                    ind[j_ - 1, i] == 0
                    and ind[j_, i] == 1
                    and ind[j_ + 1, i] == 1
                    and ind[j_ + 2, i] == 1
                ) or (
                    ind[j_ - 1, i] == 1
                    and ind[j_, i] == 1
                    and ind[j_ + 1, i] == 1
                    and ind[j_ + 2, i] == 0
                ):
                    AddMat(i + (Nx + 1) * (j_ - 1), i, j_ - 1, sigma)
                    AddMat(i + (Nx + 1) * (j_ - 1), i, j_, -3 * sigma)
                    AddMat(i + (Nx + 1) * (j_ - 1), i, j_ + 1, 3 * sigma)
                    AddMat(i + (Nx + 1) * (j_ - 1), i, j_ + 2, -sigma)
                    AddMat(i + (Nx + 1) * j_, i, j_ - 1, -3 * sigma)
                    AddMat(i + (Nx + 1) * j_, i, j_, 9 * sigma)
                    AddMat(i + (Nx + 1) * j_, i, j_ + 1, -9 * sigma)
                    AddMat(i + (Nx + 1) * j_, i, j_ + 2, 3 * sigma)
                    AddMat(i + (Nx + 1) * (j_ + 1), i, j_ - 1, 3 * sigma)
                    AddMat(i + (Nx + 1) * (j_ + 1), i, j_, -9 * sigma)
                    AddMat(i + (Nx + 1) * (j_ + 1), i, j_ + 1, 9 * sigma)
                    AddMat(i + (Nx + 1) * (j_ + 1), i, j_ + 2, -3 * sigma)
                    AddMat(i + (Nx + 1) * (j_ + 2), i, j_ - 1, -sigma)
                    AddMat(i + (Nx + 1) * (j_ + 2), i, j_, 3 * sigma)
                    AddMat(i + (Nx + 1) * (j_ + 2), i, j_ + 1, -3 * sigma)
                    AddMat(i + (Nx + 1) * (j_ + 2), i, j_ + 2, sigma)

            # penalistion of the boundary condition
            npcoef = 1.0 / (hx**2) * np.array(coef)
            C = sp.coo_array((npcoef, (row, col)), shape=(Ndof, Ndof))

            # penalization outside
            D = sp.diags(diagonals=indOut.ravel())

            # linear system
            A = (A + B + C + D).tocsr()
            b = f(X, Y)
            b = (ind * b).ravel()
            u = spsolve(A, b).reshape(Ny + 1, Nx + 1)
            e = u - uref

            eL2 = np.sqrt(np.sum(e * e * (1 - indOut)) * hx * hy) / np.sqrt(
                np.sum(uref * uref * (1 - indOut)) * hx * hy
            )
            emax = np.max(np.abs(e * (1 - indOut))) / np.max(
                np.abs(uref * (1 - indOut))
            )

            ex = (e[:, 1 : Nx + 1] - e[:, 0:Nx]) / hx
            urefx = (uref[:, 1 : Nx + 1] - uref[:, 0:Nx]) / hx
            intdx = (ind[:, 1 : Nx + 1] + ind[:, 0:Nx] == 2) + 0
            fulldx = (ind[:, 1 : Nx + 1] + ind[:, 0:Nx] > 0) + 0

            ey = (e[1 : Ny + 1, :] - e[0:Ny, :]) / hy
            urefy = (uref[1 : Ny + 1, :] - uref[0:Ny, :]) / hy
            intdy = (ind[1 : Ny + 1, :] + ind[0:Ny, :] == 2) + 0
            fulldy = (ind[1 : Ny + 1, :] + ind[0:Ny, :] > 0) + 0

            eH1 = np.sqrt(
                (np.sum(ex * ex * fulldx) + np.sum(ey * ey * fulldy)) * hx * hy
            ) / np.sqrt(
                (np.sum(urefx * urefx * fulldx) + np.sum(urefy * urefy * fulldy))
                * hx
                * hy
            )
            eH1int = np.sqrt(
                (np.sum(ex * ex * intdx) + np.sum(ey * ey * intdy)) * hx * hy
            ) / np.sqrt(
                (np.sum(urefx * urefx * intdx) + np.sum(urefy * urefy * intdy))
                * hx
                * hy
            )

            if conditioning == True:
                cond = np.linalg.cond(A.todense())
                cond_diff_vec_change_sigma.append(cond)
            error_L2_diff_vec_change_sigma.append(eL2)
            error_Linf_diff_vec_change_sigma.append(emax)
            error_H1_diff_vec_change_sigma.append(eH1)
            error_H1int_diff_vec_change_sigma.append(eH1int)
        error_L2_diff_vec_change_gamma.append(error_L2_diff_vec_change_sigma)
        error_Linf_diff_vec_change_gamma.append(error_Linf_diff_vec_change_sigma)
        error_H1_diff_vec_change_gamma.append(error_H1_diff_vec_change_sigma)
        error_H1int_diff_vec_change_gamma.append(error_H1int_diff_vec_change_sigma)
        if conditioning == True:
            cond_diff_vec_change_gamma.append(cond_diff_vec_change_sigma)
    error_L2_diff_vec_2.append(error_L2_diff_vec_change_gamma)
    error_Linf_diff_vec_2.append(error_Linf_diff_vec_change_gamma)
    error_H1_diff_vec_2.append(error_H1_diff_vec_change_gamma)
    error_H1int_diff_vec_2.append(error_H1int_diff_vec_change_gamma)
    if conditioning == True:
        cond_diff_vec_2.append(cond_diff_vec_change_gamma)
error_L2_diff_array_2 = np.array(error_L2_diff_vec_2)
error_Linf_diff_array_2 = np.array(error_Linf_diff_vec_2)
error_H1_diff_array_2 = np.array(error_H1_diff_vec_2)
error_H1int_diff_array_2 = np.array(error_H1int_diff_vec_2)
if conditioning == True:
    cond_diff_array_2 = np.array(cond_diff_vec_2)
size_mesh_array_2 = np.zeros_like(error_L2_diff_array_2)
size_mesh_array_2[:, :, :] = size_mesh[:, None, None]


if conditioning == True:
    full_results_array_2 = np.array(
        [
            size_mesh_array_2,
            error_L2_diff_array_2,
            error_Linf_diff_array_2,
            error_H1_diff_array_2,
            error_H1int_diff_array_2,
            cond_diff_array_2,
        ]
    )
    np.save("full_res_with_cond_phiFD2.npy", full_results_array_2)
else:

    full_results_array_2 = np.array(
        [
            size_mesh_array_2,
            error_L2_diff_array_2,
            error_Linf_diff_array_2,
            error_H1_diff_array_2,
            error_H1int_diff_array_2,
        ]
    )

    np.save("full_res_without_cond_phiFD2.npy", full_results_array_2)
print(f"{full_results_array_2.shape=}")

# Gamma = [0.001, 0.01, 0.1, 1.0, 10.0, 20.0]
# Sigma = [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 20.0]

index_sigma_1 = 1
index_gamma_1 = 3
index_sigma_2 = 1
index_gamma_2 = 3

if conditioning == True:
    plt.figure(figsize=(18, 12))
    linestyle_1 = ["--+", "--x", "--*", "--o", "--s", "-->"]
    linestyle_2 = ["-+", "-x", "-*", "-o", "-s", "->"]
    markers = ["+", "x", "*", "o", "s", ">"]
    lines_legend = [
        plt.Line2D([0], [0], color="k", marker=markers[i], linestyle="")
        for i in range(len(NN))
    ]
    labels = [f"$h=${size_mesh[i]:.2f}" for i in range(len(NN))]
    plt.subplot(2, 3, 1)
    for i in range(len(NN)):
        plt.loglog(
            Sigma,
            error_L2_diff_array_1[i, index_gamma_1, :],
            linestyle_1[i],
            label=f"$h=${size_mesh[i]:.2f}",
        )
        plt.loglog(
            Sigma,
            error_L2_diff_array_2[i, index_gamma_2, :],
            linestyle_2[i],
            label=f"$h=${size_mesh[i]:.2f}",
        )
    plt.title(
        f"$\phi$-FD 1: $\gamma=${Gamma[index_gamma_1]} $\phi$-FD 2: $\gamma=${Gamma[index_gamma_2]}",
        fontsize=16,
    )
    plt.xlabel(f"$\sigma$", fontsize=16)
    plt.ylabel(f"$L^2$ error", fontsize=16)
    plt.legend(handles=lines_legend, labels=labels, ncol=2, fontsize=16)
    plt.subplot(2, 3, 2)
    for i in range(len(NN)):
        plt.loglog(
            Sigma,
            error_H1int_diff_array_1[i, index_gamma_1, :],
            linestyle_1[i],
            label=f"$h=${size_mesh[i]:.2f}",
        )
        plt.loglog(
            Sigma,
            error_H1int_diff_array_2[i, index_gamma_2, :],
            linestyle_2[i],
            label=f"$h=${size_mesh[i]:.2f}",
        )

    plt.xlabel(f"$\sigma$", fontsize=16)
    plt.ylabel(f"$H^1$ error")
    plt.title(
        f"$\phi$-FD 1: $\gamma=${Gamma[index_gamma_1]} $\phi$-FD 2: $\gamma=${Gamma[index_gamma_2]}",
        fontsize=16,
    )
    plt.legend(ncol=2, fontsize=16)
    plt.subplot(2, 3, 3)
    for i in range(len(NN)):
        plt.loglog(
            Sigma,
            cond_diff_array_1[i, index_gamma_1, :],
            linestyle_1[i],
            label=f"$h=${size_mesh[i]:.2f}",
        )
        plt.loglog(
            Sigma,
            cond_diff_array_2[i, index_gamma_2, :],
            linestyle_2[i],
            label=f"$h=${size_mesh[i]:.2f}",
        )
    plt.title(
        f"$\phi$-FD 1: $\gamma=${Gamma[index_gamma_1]} $\phi$-FD 2: $\gamma=${Gamma[index_gamma_2]}",
        fontsize=16,
    )
    plt.xlabel(f"$\sigma$", fontsize=16)
    plt.ylabel(f"Condition number", fontsize=16)
    plt.legend(ncol=2, fontsize=16)

    plt.subplot(2, 3, 4)
    for i in range(len(NN)):
        plt.loglog(
            Gamma,
            error_L2_diff_array_1[i, :, index_sigma_1],
            linestyle_1[i],
            label=f"$h=${size_mesh[i]:.2f}",
        )
        plt.loglog(
            Gamma,
            error_L2_diff_array_2[i, :, index_sigma_2],
            linestyle_2[i],
            label=f"$h=${size_mesh[i]:.2f}",
        )
    plt.title(
        f"$\phi$-FD 1: $\sigma=${Sigma[index_sigma_1]} $\phi$-FD 2: $\sigma=${Sigma[index_sigma_2]}",
        fontsize=16,
    )
    plt.xlabel(f"$\gamma$", fontsize=16)
    plt.ylabel(f"$L^2$ error", fontsize=16)
    plt.legend(ncol=2, fontsize=16)
    plt.subplot(2, 3, 5)
    for i in range(len(NN)):
        plt.loglog(
            Gamma,
            error_H1int_diff_array_1[i, :, index_sigma_1],
            linestyle_1[i],
            label=f"$h=${size_mesh[i]:.2f}",
        )
        plt.loglog(
            Gamma,
            error_H1int_diff_array_2[i, :, index_sigma_2],
            linestyle_2[i],
            label=f"$h=${size_mesh[i]:.2f}",
        )
    plt.xlabel(f"$\gamma$", fontsize=16)
    plt.ylabel(f"$H^1$ error", fontsize=16)
    plt.legend(ncol=2, fontsize=16)
    plt.title(
        f"$\phi$-FD 1: $\sigma=${Sigma[index_sigma_1]} $\phi$-FD 2: $\sigma=${Sigma[index_sigma_2]}",
        fontsize=16,
    )
    plt.subplot(2, 3, 6)
    for i in range(len(NN)):
        plt.loglog(
            Gamma,
            cond_diff_array_1[i, :, index_sigma_1],
            linestyle_1[i],
            label=f"$h=${size_mesh[i]:.2f}",
        )
        plt.loglog(
            Gamma,
            cond_diff_array_2[i, :, index_sigma_2],
            linestyle_2[i],
            label=f"$h=${size_mesh[i]:.2f}",
        )

    plt.xlabel(f"$\gamma$", fontsize=16)
    plt.ylabel(f"Condition number", fontsize=16)
    plt.legend(ncol=2, fontsize=16)
    plt.title(
        f"$\phi$-FD 1: $\sigma=${Sigma[index_sigma_1]} $\phi$-FD 2: $\sigma=${Sigma[index_sigma_2]}",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig("res_with_cond.pdf")
    plt.show()


else:
    plt.figure(figsize=(12, 12))
    linestyle_1 = ["--+", "--x", "--*", "--o", "--s", "-->"]
    linestyle_2 = ["-+", "-x", "-*", "-o", "-s", "->"]
    markers = ["+", "x", "*", "o", "s", ">"]
    lines_legend = [
        plt.Line2D([0], [0], color="k", marker=markers[i], linestyle="")
        for i in range(len(NN))
    ]
    labels = [f"$h=${size_mesh[i]:.2f}" for i in range(len(NN))]

    plt.subplot(2, 2, 1)
    for i in range(len(NN)):
        plt.loglog(
            Sigma,
            error_L2_diff_array_1[i, index_gamma_1, :],
            linestyle_1[i],
            label=f"$h=${size_mesh[i]:.2f}",
        )
        plt.loglog(
            Sigma,
            error_L2_diff_array_2[i, index_gamma_2, :],
            linestyle_2[i],
            label=f"$h=${size_mesh[i]:.2f}",
        )
    plt.title(
        f"$\phi$-FD 1: $\gamma=${Gamma[index_gamma_1]} $\phi$-FD 2: $\gamma=${Gamma[index_gamma_2]}",
        fontsize=16,
    )
    plt.xlabel(f"$\sigma$", fontsize=16)
    plt.ylabel(f"$L^2$ error", fontsize=16)
    plt.legend(handles=lines_legend, labels=labels, ncol=2, fontsize=16)
    # plt.legend(ncol=2, fontsize=16)
    plt.subplot(2, 2, 2)
    for i in range(len(NN)):
        plt.loglog(
            Sigma,
            error_H1int_diff_array_1[i, index_gamma_1, :],
            linestyle_1[i],
            label=f"$h=${size_mesh[i]:.2f}",
        )
        plt.loglog(
            Sigma,
            error_H1int_diff_array_2[i, index_gamma_2, :],
            linestyle_2[i],
            label=f"$h=${size_mesh[i]:.2f}",
        )

    plt.xlabel(f"$\sigma$", fontsize=16)
    plt.ylabel(f"$H^1$ error")
    plt.title(
        f"$\phi$-FD 1: $\gamma=${Gamma[index_gamma_1]} $\phi$-FD 2: $\gamma=${Gamma[index_gamma_2]}",
        fontsize=16,
    )
    plt.legend(handles=lines_legend, labels=labels, ncol=2, fontsize=16)
    # plt.legend(ncol=2, fontsize=16)
    plt.subplot(2, 2, 3)
    for i in range(len(NN)):
        plt.loglog(
            Gamma,
            error_L2_diff_array_1[i, :, index_sigma_1],
            linestyle_1[i],
            label=f"$h=${size_mesh[i]:.2f}",
        )
        plt.loglog(
            Gamma,
            error_L2_diff_array_2[i, :, index_sigma_2],
            linestyle_2[i],
            label=f"$h=${size_mesh[i]:.2f}",
        )
    plt.title(
        f"$\phi$-FD 1: $\sigma=${Sigma[index_sigma_1]} $\phi$-FD 2: $\sigma=${Sigma[index_sigma_2]}",
        fontsize=16,
    )
    plt.xlabel(f"$\gamma$", fontsize=16)
    plt.ylabel(f"$L^2$ error", fontsize=16)
    plt.legend(handles=lines_legend, labels=labels, ncol=2, fontsize=16)
    # plt.legend(ncol=2, fontsize=16)
    plt.subplot(2, 2, 4)
    for i in range(len(NN)):
        plt.loglog(
            Gamma,
            error_H1int_diff_array_1[i, :, index_sigma_1],
            linestyle_1[i],
            label=f"$h=${size_mesh[i]:.2f}",
        )
        plt.loglog(
            Gamma,
            error_H1int_diff_array_2[i, :, index_sigma_2],
            linestyle_2[i],
            label=f"$h=${size_mesh[i]:.2f}",
        )
    plt.xlabel(f"$\gamma$", fontsize=16)
    plt.ylabel(f"$H^1$ error", fontsize=16)
    plt.legend(handles=lines_legend, labels=labels, ncol=2, fontsize=16)
    # plt.legend(ncol=2, fontsize=16)
    plt.title(
        f"$\phi$-FD 1: $\sigma=${Sigma[index_sigma_1]} $\phi$-FD 2: $\sigma=${Sigma[index_sigma_2]}",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig("res_without_cond.pdf")
    plt.show()
