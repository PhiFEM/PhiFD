from __future__ import print_function
import numpy as np
import dolfin as df
import sympy
import matplotlib.pyplot as plt
from time import time
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import pylab as py
import platform
import mshr

import seaborn as sns

sns.set_theme("paper")

print("version python", platform.python_version())

df.parameters["ghost_mode"] = "shared_facet"
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True
df.parameters["allow_extrapolation"] = True
df.parameters["form_compiler"]["representation"] = "uflacs"

# radius of the domains
R_x = 0.3 + 1e-10
R_y = 0.3 + 1e-10

# Polynome Pk
polV = 1
polPhi = polV +1 
# parameters["form_compiler"]["quadrature_degree"]=2*(polV+polPhi)

# Ghost penalty
ghost = True

# plot the solution
Plot = False

# Compute the conditioning number
conditioning = True
# Number of iterations
init_Iter = 1
Iter = 4 if conditioning else 6


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

###########################################
### beginning phi fem ###########
###########################################

# parameter of the ghost penalty
sigma = 1.0

# Initialistion of the output
size_mesh_phi_vec = np.zeros(Iter)
error_L2_phifem_vec = np.zeros(Iter)
error_Linf_phifem_vec = np.zeros(Iter)
error_H1_phifem_vec = np.zeros(Iter)
error_H1int_phifem_vec = np.zeros(Iter)
cond_phifem_vec = np.zeros(Iter)
time_phifem_vec = np.zeros(Iter)
for i in range(init_Iter - 1, Iter):
    print("#########################")
    print("## Iteration phifem ", i + 1, "##")
    print("#########################")

    # Construction of the mesh
    N = int(10 * 2 ** ((i)))
    print("N=", N)
    mesh_macro = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(1.0, 1.0), N, N)
    domains = df.MeshFunction("size_t", mesh_macro, mesh_macro.topology().dim())
    domains.set_all(0)
    for ind in range(mesh_macro.num_cells()):
        mycell = df.Cell(mesh_macro, ind)
        v1x, v1y, v2x, v2y, v3x, v3y = mycell.get_vertex_coordinates()
        if Omega(v1x, v1y) or Omega(v2x, v2y) or Omega(v3x, v3y):
            domains[ind] = 1
    mesh = df.SubMesh(mesh_macro, domains, 1)
    print("num cells:", mesh.num_cells())
    V = df.FunctionSpace(mesh, "CG", polV)
    V_phi = df.FunctionSpace(mesh, "CG", polPhi)

    # Construction of phi
    phi = df.Expression(
        "-1.0 + pow((x[0]-0.5)/R_x,2)+pow((x[1]-0.5)/R_y,2)",
        R_x=R_x,
        R_y=R_y,
        degree=polPhi,
        domain=mesh,
    )
    phi = df.interpolate(phi, V_phi)

    # Computation of the source term
    f_expr = df.Expression(
        sympy.ccode(f1).replace("xx", "x[0]").replace("yy", "x[1]"),
        degree=polV,
        domain=mesh,
    )
    u_expr = df.Expression(
        sympy.ccode(u1).replace("xx", "x[0]").replace("yy", "x[1]"),
        degree=4,
        domain=mesh,
    )
    g = df.Expression("0.0", degree=polV + 2, domain=mesh)

    # Facets and cells where we apply the ghost penalty
    mesh.init(1, 2)
    facet_ghost = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    cell_ghost = df.MeshFunction("size_t", mesh, mesh.topology().dim())
    facet_ghost.set_all(0)
    cell_ghost.set_all(0)
    count_cell_ghost = 0
    for mycell in df.cells(mesh):
        for myfacet in df.facets(mycell):
            v1, v2 = df.vertices(myfacet)
            if (
                phi(v1.point().x(), v1.point().y())
                * phi(v2.point().x(), v2.point().y())
                < df.DOLFIN_EPS
            ):
                cell_ghost[mycell] = 1
                for myfacet2 in df.facets(mycell):
                    facet_ghost[myfacet2] = 1

    for mycell in df.cells(mesh):
        if cell_ghost[mycell] == 1:
            count_cell_ghost += 1
    print("num of cell in the ghost penalty:", count_cell_ghost)

    t_init = time()
    # Initialize cell function for domains
    dx = df.Measure("dx")(domain=mesh, subdomain_data=cell_ghost)
    ds = df.Measure("ds")(domain=mesh)
    dS = df.Measure("dS")(domain=mesh, subdomain_data=facet_ghost)

    # Resolution
    n = df.FacetNormal(mesh)
    h = df.CellDiameter(mesh)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)

    a = (
        df.inner(df.grad(phi * u), df.grad(phi * v)) * dx
        - df.dot(df.inner(df.grad(phi * u), n), phi * v) * ds
        + sigma
        * df.avg(h)
        * df.dot(
            df.jump(df.grad(phi * u), n),
            df.jump(df.grad(phi * v), n),
        )
        * dS(1)
        + sigma
        * h**2
        * df.inner(
            df.div(df.grad(phi * u)),
            df.div(df.grad(phi * v)),
        )
        * dx(1)
    )
    L = f_expr * v * phi * dx - sigma * h**2 * df.inner(
        f_expr, df.div(df.grad(phi * v))
    ) * dx(1)

    # computation of the solution
    w_h = df.Function(V)
    print("ready o solve")
    df.solve(a == L, w_h)  # , solver_parameters={'linear_solver': 'mumps'})
    sol = df.project(w_h * phi + g, V)
    t_final = time()

    # computation of the error
    V_macro = df.FunctionSpace(mesh_macro, "CG", 1)
    sol2 = df.interpolate(sol, V_macro)
    Nx = N
    Ny = N
    x = np.linspace(0.0, 1.0, Nx + 1)
    y = np.linspace(0.0, 1.0, Ny + 1)

    phi = lambda x, y: -1.0 + ((x - 0.5) / R_x) ** 2 + ((y - 0.5) / R_y) ** 2

    Ndof = (Nx + 1) * (Ny + 1)
    hx = x[1] - x[0]
    hy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)  # 2D meshgrid

    sol_values = np.zeros((Nx + 1, Ny + 1))
    for ix in range(Nx + 1):
        for iy in range(Ny + 1):
            sol_values[iy, ix] = sol2(x[ix], y[iy])

    # print(sol_values[10,10],sol2(10/Nx,10/Ny))
    phiij = phi(X, Y)
    ind = (phiij < 0) + 0
    indOut = 1 - ind
    ue = sympy.lambdify([x_symb, y_symb], u1)
    uref = ue(X, Y)

    e = sol_values - uref
    # print(type(inf))

    eL2 = np.sqrt(np.sum(e * e * (1 - indOut)) * hx * hy) / np.sqrt(
        np.sum(uref * uref * (1 - indOut)) * hx * hy
    )
    emax = np.max(np.abs(e * (1 - indOut))) / np.max(np.abs(uref * (1 - indOut)))

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
        (np.sum(urefx * urefx * fulldx) + np.sum(urefy * urefy * fulldy)) * hx * hy
    )
    eH1int = np.sqrt(
        (np.sum(ex * ex * intdx) + np.sum(ey * ey * intdy)) * hx * hy
    ) / np.sqrt(
        (np.sum(urefx * urefx * intdx) + np.sum(urefy * urefy * intdy)) * hx * hy
    )

    print("h :", mesh.hmax())
    print("relative L2 error diff SW : ", eL2)
    print("relative L inf error diff SW : ", emax)
    print("relative H1 int error diff SW : ", eH1int)
    print("time : ", t_final - t_init)
    if conditioning == True:
        A = np.matrix(df.assemble(a).array())
        cond = np.linalg.cond(A)
        cond_phifem_vec[i] = cond
        print("conditioning number diff SW : ", cond)
    time_phifem_vec[i] = t_final - t_init
    error_L2_phifem_vec[i] = eL2
    error_Linf_phifem_vec[i] = emax
    error_H1_phifem_vec[i] = eH1
    error_H1int_phifem_vec[i] = eH1int
    size_mesh_phi_vec[i] = mesh.hmax()

###########################################
### beginning standard fem ###########
###########################################

# Initialistion of the output
size_mesh_standard_vec = np.zeros(Iter)
error_L2_standard_vec = np.zeros(Iter)
error_Linf_standard_vec = np.zeros(Iter)
error_H1_standard_vec = np.zeros(Iter)
error_H1int_standard_vec = np.zeros(Iter)
cond_standard_vec = np.zeros(Iter)
time_standard_vec = np.zeros(Iter)
domain = mshr.Ellipse(df.Point(0.5, 0.5), R_x, R_y)  # creation of the domain
for i in range(init_Iter - 1, Iter):
    print("#########################")
    print("## Iteration standard ", i + 1, "##")
    print("#########################")

    t_init = time()
    # Construction of the mesh
    H = 11 * 2 ** (i - 1)
    mesh = mshr.generate_mesh(domain, H)

    print("num cells:", mesh.num_cells())
    V = df.FunctionSpace(mesh, "CG", polV)

    # Computation of the source term
    f_expr = df.Expression(
        sympy.ccode(f1).replace("xx", "x[0]").replace("yy", "x[1]"),
        degree=polV,
        domain=mesh,
    )
    u_expr = df.Expression(
        sympy.ccode(u1).replace("xx", "x[0]").replace("yy", "x[1]"),
        degree=4,
        domain=mesh,
    )

    g = df.Expression("0.0", degree=polV + 2, domain=mesh)

    # Initialize cell function for domains
    dx = df.Measure("dx")(domain=mesh)
    ds = df.Measure("ds")(domain=mesh)
    dS = df.Measure("dS")(domain=mesh)

    # Resolution
    h = df.CellDiameter(mesh)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    a = df.inner(df.grad(u), df.grad(v)) * dx
    L = f_expr * v * dx

    # Define boundary condition
    def boundary(x, on_boundary):
        return on_boundary

    bc = df.DirichletBC(V, g, boundary)

    # Define solution function
    sol = df.Function(V)
    print("ready to solve")
    df.solve(a == L, sol, bc)  # , solver_parameters={'linear_solver': 'mumps'})
    print("solved")
    t_final = time()

    # Computation of the error
    N = int(10 * 2 ** ((i)))
    print("N=", N)
    mesh_macro = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(1.0, 1.0), N, N)
    V_macro = df.FunctionSpace(mesh_macro, "CG", 1)
    sol2 = df.interpolate(sol, V_macro)
    Nx = N
    Ny = N
    x = np.linspace(0.0, 1.0, Nx + 1)
    y = np.linspace(0.0, 1.0, Ny + 1)

    phi = lambda x, y: -1.0 + ((x - 0.5) / R_x) ** 2 + ((y - 0.5) / R_y) ** 2

    Ndof = (Nx + 1) * (Ny + 1)
    hx = x[1] - x[0]
    hy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)  # 2D meshgrid

    sol_values = np.zeros((Nx + 1, Ny + 1))
    for ix in range(Nx + 1):
        for iy in range(Ny + 1):
            sol_values[iy, ix] = sol2(x[ix], y[iy])

    phiij = phi(X, Y)
    ind = (phiij < 0) + 0
    indOut = 1 - ind
    ue = sympy.lambdify([x_symb, y_symb], u1)
    uref = ue(X, Y)

    e = sol_values - uref
    # print(type(inf))

    eL2 = np.sqrt(np.sum(e * e * (1 - indOut)) * hx * hy) / np.sqrt(
        np.sum(uref * uref * (1 - indOut)) * hx * hy
    )
    emax = np.max(np.abs(e * (1 - indOut))) / np.max(np.abs(uref * (1 - indOut)))

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
        (np.sum(urefx * urefx * fulldx) + np.sum(urefy * urefy * fulldy)) * hx * hy
    )
    eH1int = np.sqrt(
        (np.sum(ex * ex * intdx) + np.sum(ey * ey * intdy)) * hx * hy
    ) / np.sqrt(
        (np.sum(urefx * urefx * intdx) + np.sum(urefy * urefy * intdy)) * hx * hy
    )

    print("h :", mesh.hmax())
    print("relative L2 error standard : ", eL2)
    print("relative L inf error standard : ", emax)
    print("relative H1 int error standard : ", eH1int)
    print("time : ", t_final - t_init)
    if conditioning == True:
        A = np.matrix(df.assemble(a).array())
        cond = np.linalg.cond(A)
        cond_standard_vec[i] = cond
        print("conditioning number standard : ", cond)
    time_standard_vec[i] = t_final - t_init
    error_L2_standard_vec[i] = eL2
    error_Linf_standard_vec[i] = emax
    error_H1_standard_vec[i] = eH1
    error_H1int_standard_vec[i] = eH1int
    size_mesh_standard_vec[i] = mesh.hmax()

###########################################
### beginning finite difference shortley welley ###########
###########################################


# Initialistion of the output
interp_errors_SW = np.zeros(Iter)
size_mesh_diff_SW_vec = np.zeros(Iter)
error_L2_diff_SW_vec = np.zeros(Iter)
error_Linf_diff_SW_vec = np.zeros(Iter)
error_H1_diff_SW_vec = np.zeros(Iter)
error_H1int_diff_SW_vec = np.zeros(Iter)
cond_diff_SW_vec = np.zeros(Iter)
time_diff_SW_vec = np.zeros(Iter)
for ii in range(init_Iter - 1, Iter):
    print("######################")
    print("## Iteration diff shortley welley", ii + 1, "##")
    print("######################")

    # begin time
    t_init = time()

    # Construction of the mesh
    N = int(10 * 2 ** ((ii)))
    print("size cells : ", np.sqrt(2) / N)
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

    # laplacian matrix

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

    # boundary correction

    # coefficients for the matrix multiplied by gamma
    diag = np.zeros((Nx + 1) * (Ny + 1))
    diagxp = np.zeros((Nx + 1) * (Ny + 1) - 1)
    diagxm = np.zeros((Nx + 1) * (Ny + 1) - 1)
    diagyp = np.zeros((Nx + 1) * Ny)
    diagym = np.zeros((Nx + 1) * Ny)

    indx = ind[:, 1 : Nx + 1] - ind[:, 0:Nx]
    J, I = np.where((indx == 1) | (indx == -1))
    for k in range(np.shape(I)[0]):
        if indx[J[k], I[k]] == 1:
            i = I[k] + 1
            j = J[k]
            hminus = hx / (1 - phiij[j, i - 1] / phiij[j, i])
            hplus = hx
        else:
            i = I[k]
            j = J[k]
            hminus = hx
            hplus = hx / (1 - phiij[j, i + 1] / phiij[j, i])
        diag[i + (Nx + 1) * j] = -2 / hx / hx + 2 / hplus / hminus
        diagxp[i + (Nx + 1) * j] = 1 / hx / hx - 2 / hplus / (hplus + hminus)
        diagxm[i - 1 + (Nx + 1) * j] = 1 / hx / hx - 2 / hminus / (hplus + hminus)

    indy = ind[1 : Ny + 1, :] - ind[0:Ny, :]
    J, I = np.where((indy == 1) | (indy == -1))
    for k in range(np.shape(I)[0]):
        if indy[J[k], I[k]] == 1:
            i = I[k]
            j = J[k] + 1
            hminus = hy / (1 - phiij[j - 1, i] / phiij[j, i])
            hplus = hy
        else:
            i = I[k]
            j = J[k]
            hminus = hy
            hplus = hy / (1 - phiij[j + 1, i] / phiij[j, i])
        diag[i + (Nx + 1) * j] += -2 / hy / hy + 2 / hplus / hminus
        diagyp[i + (Nx + 1) * j] = 1 / hy / hy - 2 / hplus / (hplus + hminus)
        diagym[i + (Nx + 1) * (j - 1)] = 1 / hy / hy - 2 / hminus / (hplus + hminus)

    B = sp.diags(
        diagonals=(diagym, diagxm, diag, diagxp, diagyp),
        offsets=(-Nx - 1, -1, 0, 1, Nx + 1),
    )

    # penalization outside
    D = sp.diags(diagonals=indOut.ravel())

    # linear system
    A = (A + B + D).tocsr()
    b = f(X, Y)
    b = (ind * b).ravel()
    u = spsolve(A, b).reshape(Ny + 1, Nx + 1)
    # final time
    t_final = time()
    uref = ue(X, Y)
    if Plot == True:
        py.contourf(x, y, u * ind, 41, cmap="viridis")
        py.colorbar()
        py.xlabel("x")
        py.ylabel("y")
        py.title("u " + str(Nx) + "X" + str(Ny))
        py.show()
        py.contourf(x, y, ind * uref, 41, cmap="viridis")
        py.colorbar()
        py.xlabel("x")
        py.ylabel("y")
        py.title("u exact")

    e = u - uref

    eL2 = np.sqrt(np.sum(e * e * (1 - indOut)) * hx * hy) / np.sqrt(
        np.sum(uref * uref * (1 - indOut)) * hx * hy
    )
    emax = np.max(np.abs(e * (1 - indOut))) / np.max(np.abs(uref * (1 - indOut)))

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
        (np.sum(urefx * urefx * fulldx) + np.sum(urefy * urefy * fulldy)) * hx * hy
    )
    eH1int = np.sqrt(
        (np.sum(ex * ex * intdx) + np.sum(ey * ey * intdy)) * hx * hy
    ) / np.sqrt(
        (np.sum(urefx * urefx * intdx) + np.sum(urefy * urefy * intdy)) * hx * hy
    )

    print("h :", np.sqrt(2) / N)
    print("relative L2 error diff SW : ", eL2)
    print("relative L inf error diff SW : ", emax)
    print("relative H1 int error diff SW : ", eH1int)
    print("time : ", t_final - t_init)
    if conditioning == True:
        cond = np.linalg.cond(A.todense())
        cond_diff_SW_vec[ii] = cond
        print("conditioning number diff SW : ", cond)
    time_diff_SW_vec[ii] = t_final - t_init
    error_L2_diff_SW_vec[ii] = eL2
    error_Linf_diff_SW_vec[ii] = emax
    error_H1int_diff_SW_vec[ii] = eH1int
    error_H1_diff_SW_vec[ii] = eH1
    size_mesh_diff_SW_vec[ii] = np.sqrt(2) / N


###########################################
### beginning finite difference ###########
###########################################
sigma = 0.01
gamma = 1.0

# Initialistion of the output
interp_errors = np.zeros(Iter)
size_mesh_diff_vec = np.zeros(Iter)
error_L2_diff_vec = np.zeros(Iter)
error_Linf_diff_vec = np.zeros(Iter)
error_H1_diff_vec = np.zeros(Iter)
error_H1int_diff_vec = np.zeros(Iter)
cond_diff_vec = np.zeros(Iter)
time_diff_vec = np.zeros(Iter)
for ii in range(init_Iter - 1, Iter):
    print("######################")
    print("## Iteration diff ", ii + 1, "##")
    print("######################")

    # begin time
    t_init = time()

    # Construction of the mesh
    N = int(10 * 2 ** ((ii)))
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

    # laplacian matrix

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
    # will create the matrix in COOrdinate format, i.e. the triplets (row,col,coef)
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

    npcoef = (gamma / hx / hy) * np.array(coef)
    B = sp.coo_array((npcoef, (row, col)), shape=(Ndof, Ndof))

    # ghost penalty
    maskGx = sp.diags(diagonals=actGx.ravel())
    maskGy = sp.diags(diagonals=actGy.ravel())
    C = sigma * hx * hy * (D2x_2d.T @ maskGx @ D2x_2d + D2y_2d.T @ maskGy @ D2y_2d)
    # penalization outside
    D = sp.diags(diagonals=indOut.ravel())

    # linear system
    A = (A + B + C + D).tocsr()
    b = f(X, Y)
    b = (ind * b).ravel()
    u = spsolve(A, b).reshape(Ny + 1, Nx + 1)
    # final time
    t_final = time()
    uref = ue(X, Y)
    if Plot == True:
        py.contourf(x, y, u * ind, 41, cmap="viridis")
        py.colorbar()
        py.xlabel("x")
        py.ylabel("y")
        py.title("u " + str(Nx) + "X" + str(Ny))
        py.show()
        py.contourf(x, y, ind * uref, 41, cmap="viridis")
        py.colorbar()
        py.xlabel("x")
        py.ylabel("y")
        py.title("u exact")

    e = u - uref

    eL2 = np.sqrt(np.sum(e * e * (1 - indOut)) * hx * hy) / np.sqrt(
        np.sum(uref * uref * (1 - indOut)) * hx * hy
    )
    emax = np.max(np.abs(e * (1 - indOut))) / np.max(np.abs(uref * (1 - indOut)))

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
        (np.sum(urefx * urefx * fulldx) + np.sum(urefy * urefy * fulldy)) * hx * hy
    )
    eH1int = np.sqrt(
        (np.sum(ex * ex * intdx) + np.sum(ey * ey * intdy)) * hx * hy
    ) / np.sqrt(
        (np.sum(urefx * urefx * intdx) + np.sum(urefy * urefy * intdy)) * hx * hy
    )

    print("h :", np.sqrt(2) / N)
    print("relative L2 error diff : ", eL2)
    print("relative L inf error diff : ", emax)
    print("relative H1 error diff : ", eH1)
    print("relative H1 error int diff : ", eH1int)
    print("time : ", t_final - t_init)
    if conditioning == True:
        cond = np.linalg.cond(A.todense())
        cond_diff_vec[ii] = cond
        print("conditioning number diff : ", cond)
    time_diff_vec[ii] = t_final - t_init
    error_L2_diff_vec[ii] = eL2
    error_Linf_diff_vec[ii] = emax
    error_H1_diff_vec[ii] = eH1
    error_H1int_diff_vec[ii] = eH1int
    size_mesh_diff_vec[ii] = np.sqrt(2) / N


###########################################
### beginning finite difference 2 ###########
###########################################
sigma = 1.0
gamma = 10.0

# Initialistion of the output
interp_errors = np.zeros(Iter)
size_mesh_diff2_vec = np.zeros(Iter)
error_L2_diff2_vec = np.zeros(Iter)
error_Linf_diff2_vec = np.zeros(Iter)
error_H1_diff2_vec = np.zeros(Iter)
error_H1int_diff2_vec = np.zeros(Iter)
cond_diff2_vec = np.zeros(Iter)
time_diff2_vec = np.zeros(Iter)
for iii in range(init_Iter - 1, Iter):
    print("######################")
    print("## Iteration diff 2 ", iii + 1, "##")
    print("######################")

    # begin time
    t_init = time()

    # Construction of the mesh
    N = int(10 * 2 ** ((iii)))
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

    # laplacian matrix

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
    # will create the matrix in COOrdinate format, i.e. the triplets (row,col,coef)
    row = []
    col = []
    coef = []  # for the matrix implementing BC

    def rav(i, j):
        return np.ravel_multi_index([j, i], (Ny + 1, Nx + 1))

    def AddMat(eq, i, j, a):
        row.append(eq)
        col.append(rav(i, j))
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
        AddMat(rav(i - 1, j), i - 1, j, aim1 * aim1 / phiS)
        AddMat(rav(i - 1, j), i, j, aim1 * ai / phiS)
        AddMat(rav(i - 1, j), i + 1, j, aim1 * aip1 / phiS)
        AddMat(rav(i, j), i - 1, j, ai * aim1 / phiS)
        AddMat(rav(i, j), i, j, ai * ai / phiS)
        AddMat(rav(i, j), i + 1, j, ai * aip1 / phiS)
        AddMat(rav(i + 1, j), i - 1, j, aip1 * aim1 / phiS)
        AddMat(rav(i + 1, j), i, j, aip1 * ai / phiS)
        AddMat(rav(i + 1, j), i + 1, j, aip1 * aip1 / phiS)

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
        AddMat(rav(i, j - 1), i, j - 1, aim1 * aim1 / phiS)
        AddMat(rav(i, j - 1), i, j, aim1 * ai / phiS)
        AddMat(rav(i, j - 1), i, j + 1, aim1 * aip1 / phiS)
        AddMat(rav(i, j), i, j - 1, ai * aim1 / phiS)
        AddMat(rav(i, j), i, j, ai * ai / phiS)
        AddMat(rav(i, j), i, j + 1, ai * aip1 / phiS)
        AddMat(rav(i, j + 1), i, j - 1, aip1 * aim1 / phiS)
        AddMat(rav(i, j + 1), i, j, aip1 * ai / phiS)
        AddMat(rav(i, j + 1), i, j + 1, aip1 * aip1 / phiS)

    # penalistion of the boundary condition
    npcoef = (gamma / hx / hy) * np.array(coef)
    B = sp.coo_array((npcoef, (row, col)), shape=(Ndof, Ndof))

    row = []
    col = []
    coef = []  # for the matrix implementing BC

    def rav(i, j):
        return np.ravel_multi_index([j, i], (Ny + 1, Nx + 1))

    def AddMat(eq, i, j, a):
        row.append(eq)
        col.append(rav(i, j))
        coef.append(a)

    indx = ind[:, 1 : Nx + 1] - ind[:, 0:Nx]
    J, I = np.where((indx == 1) | (indx == -1))
    for k in range(np.shape(I)[0]):
        if indx[J[k], I[k]] == 1:
            ii, j = I[k] + 1, J[k]
        else:
            ii, j = I[k], J[k]
        if (
            ind[j, ii - 1] == 0
            and ind[j, ii] == 1
            and ind[j, ii + 1] == 1
            and ind[j, ii + 2] == 1
        ) or (
            ind[j, ii - 1] == 1
            and ind[j, ii] == 1
            and ind[j, ii + 1] == 1
            and ind[j, ii + 2] == 0
        ):
            AddMat(rav(ii - 1, j), ii - 1, j, 1.0)
            AddMat(rav(ii - 1, j), ii, j, -3.0)
            AddMat(rav(ii - 1, j), ii + 1, j, 3.0)
            AddMat(rav(ii - 1, j), ii + 2, j, -1.0)
            AddMat(rav(ii, j), ii - 1, j, -3.0)
            AddMat(rav(ii, j), ii, j, 9.0)
            AddMat(rav(ii, j), ii + 1, j, -9.0)
            AddMat(rav(ii, j), ii + 2, j, 3.0)
            AddMat(rav(ii + 1, j), ii - 1, j, 3.0)
            AddMat(rav(ii + 1, j), ii, j, -9.0)
            AddMat(rav(ii + 1, j), ii + 1, j, 9.0)
            AddMat(rav(ii + 1, j), ii + 2, j, -3.0)
            AddMat(rav(ii + 2, j), ii - 1, j, -1.0)
            AddMat(rav(ii + 2, j), ii, j, 3.0)
            AddMat(rav(ii + 2, j), ii + 1, j, -3.0)
            AddMat(rav(ii + 2, j), ii + 2, j, 1.0)

    indy = ind[1 : Ny + 1, :] - ind[0:Ny, :]
    J, I = np.where((indy == 1) | (indy == -1))
    for k in range(np.shape(I)[0]):
        if indy[J[k], I[k]] == 1:
            i, jj = I[k], J[k] + 1
        else:
            i, jj = I[k], J[k]
        if (
            ind[jj - 1, i] == 0
            and ind[jj, i] == 1
            and ind[jj + 1, i] == 1
            and ind[jj + 2, i] == 1
        ) or (
            ind[jj - 1, i] == 1
            and ind[jj, i] == 1
            and ind[jj + 1, i] == 1
            and ind[jj + 2, i] == 0
        ):
            AddMat(rav(i, jj - 1), i, jj - 1, 1.0)
            AddMat(rav(i, jj - 1), i, jj, -3.0)
            AddMat(rav(i, jj - 1), i, jj + 1, 3.0)
            AddMat(rav(i, jj - 1), i, jj + 2, -1.0)
            AddMat(rav(i, jj), i, jj - 1, -3.0)
            AddMat(rav(i, jj), i, jj, 9.0)
            AddMat(rav(i, jj), i, jj + 1, -9.0)
            AddMat(rav(i, jj), i, jj + 2, 3.0)
            AddMat(rav(i, jj + 1), i, jj - 1, 3.0)
            AddMat(rav(i, jj + 1), i, jj, -9.0)
            AddMat(rav(i, jj + 1), i, jj + 1, 9.0)
            AddMat(rav(i, jj + 1), i, jj + 2, -3.0)
            AddMat(rav(i, jj + 2), i, jj - 1, -1.0)
            AddMat(rav(i, jj + 2), i, jj, 3.0)
            AddMat(rav(i, jj + 2), i, jj + 1, -3.0)
            AddMat(rav(i, jj + 2), i, jj + 2, 1.0)

    # penalistion of the boundary condition
    npcoef = np.array(coef) * (sigma / hx / hy)
    C = sp.coo_array((npcoef, (row, col)), shape=(Ndof, Ndof))

    # penalization outside
    D = sp.diags(diagonals=indOut.ravel())

    # linear system
    A = (A + B + C + D).tocsr()
    b = f(X, Y)
    b = (ind * b).ravel()
    u = spsolve(A, b).reshape(Ny + 1, Nx + 1)
    # final time
    t_final = time()
    uref = ue(X, Y)
    if Plot == True:
        py.contourf(x, y, u * ind, 41, cmap="viridis")
        py.colorbar()
        py.xlabel("x")
        py.ylabel("y")
        py.title("u " + str(Nx) + "X" + str(Ny))
        py.show()
        py.contourf(x, y, ind * uref, 41, cmap="viridis")
        py.colorbar()
        py.xlabel("x")
        py.ylabel("y")
        py.title("u exact")

    e = u - uref
    eL2 = np.sqrt(np.sum(e * e * (1 - indOut)) * hx * hy) / np.sqrt(
        np.sum(uref * uref * (1 - indOut)) * hx * hy
    )
    emax = np.max(np.abs(e * (1 - indOut))) / np.max(np.abs(uref * (1 - indOut)))
    # print("errors full : ", eL2, emax)

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
        (np.sum(urefx * urefx * fulldx) + np.sum(urefy * urefy * fulldy)) * hx * hy
    )
    eH1int = np.sqrt(
        (np.sum(ex * ex * intdx) + np.sum(ey * ey * intdy)) * hx * hy
    ) / np.sqrt(
        (np.sum(urefx * urefx * intdx) + np.sum(urefy * urefy * intdy)) * hx * hy
    )

    print("h :", np.sqrt(2) / N)
    print("relative L2 error diff : ", eL2)
    # print("relative L2 error diff bis : ", eL2_bis)
    print("relative L inf error diff : ", emax)
    print("relative H1 error diff : ", eH1)
    print("relative H1 error diff int : ", eH1int)
    # print("relative L inf error diff bis : ", emax_bis)
    print("time : ", t_final - t_init)
    if conditioning == True:
        cond = np.linalg.cond(A.todense())
        cond_diff2_vec[iii] = cond
        print("conditioning number diff : ", cond)
    time_diff2_vec[iii] = t_final - t_init
    error_L2_diff2_vec[iii] = eL2
    error_Linf_diff2_vec[iii] = emax
    error_H1_diff2_vec[iii] = eH1
    error_H1int_diff2_vec[iii] = eH1int
    size_mesh_diff2_vec[iii] = np.sqrt(2) / N


# # Print the output vectors
print("Vector h :", size_mesh_phi_vec)
print("Vector relative L2 error phifem : ", error_L2_phifem_vec)
print("Vector relative Linf error phifem : ", error_Linf_phifem_vec)
print("Vector relative H1 int int error phifem : ", error_H1int_phifem_vec)
print("Vector time phifem : ", time_phifem_vec)
if conditioning == True:
    print("conditioning number phifem :", cond_phifem_vec)
print("Vector relative L2 error standard : ", error_L2_standard_vec)
print("Vector relative Linf error standard : ", error_Linf_standard_vec)
print("Vector relative H1 int error standard : ", error_H1int_standard_vec)
print("Vector time standard : ", time_standard_vec)
if conditioning == True:
    print("conditioning number standard :", cond_standard_vec)
print("Vector relative L2 error diff SW : ", error_L2_diff_SW_vec)
print("Vector relative Linf error diff SW : ", error_Linf_diff_SW_vec)
print("Vector relative H1 int error diff SW : ", error_H1int_diff_SW_vec)
print("Vector time diff SW : ", time_diff_SW_vec)
if conditioning == True:
    print("conditioning number diff SW :", cond_diff_SW_vec)
print("Vector relative L2 error diff : ", error_L2_diff_vec)
print("Vector relative Linf error diff : ", error_Linf_diff_vec)
print("Vector relative H1 int error diff : ", error_H1int_diff_vec)
print("Vector time diff : ", time_diff_vec)
if conditioning == True:
    print("conditioning number diff :", cond_diff_vec)
print("Vector relative L2 error diff2 : ", error_L2_diff2_vec)
print("Vector relative Linf error diff2 : ", error_Linf_diff2_vec)
print("Vector relative H1 int error diff2 : ", error_H1int_diff2_vec)
print("Vector time diff2 : ", time_diff2_vec)
if conditioning == True:
    print("conditioning number diff2 :", cond_diff2_vec)


def order(a, b):
    return -np.polyfit(np.log(a), -np.log(b), 1)[0]


order_L2_phifem = order(size_mesh_phi_vec, error_L2_phifem_vec)
order_Linf_phifem = order(size_mesh_phi_vec, error_Linf_phifem_vec)
order_H1int_phifem = order(size_mesh_phi_vec, error_H1int_phifem_vec)
order_L2_standard = order(size_mesh_standard_vec, error_L2_standard_vec)
order_Linf_standard = order(size_mesh_standard_vec, error_Linf_standard_vec)
order_H1int_standard = order(size_mesh_standard_vec, error_H1int_standard_vec)
order_L2_diff_SW = order(size_mesh_diff_SW_vec, error_L2_diff_SW_vec)
order_Linf_diff_SW = order(size_mesh_diff_SW_vec, error_Linf_diff_SW_vec)
order_H1int_diff_SW = order(size_mesh_diff_SW_vec, error_H1int_diff_SW_vec)
order_L2_diff = order(size_mesh_diff_vec, error_L2_diff_vec)
order_Linf_diff = order(size_mesh_diff_vec, error_Linf_diff_vec)
order_H1int_diff = order(size_mesh_diff_vec, error_H1int_diff_vec)
order_L2_diff2 = order(size_mesh_diff2_vec, error_L2_diff2_vec)
order_Linf_diff2 = order(size_mesh_diff2_vec, error_Linf_diff2_vec)
order_H1int_diff2 = order(size_mesh_diff2_vec, error_H1int_diff2_vec)


print("Order conv rel L2 error phifem : ", order_L2_phifem)
print("Order conv rel Linf error phifem : ", order_Linf_phifem)
print("Order conv rel H1 error phifem : ", order_H1int_phifem)

print("Order conv rel L2 error standard : ", order_L2_standard)
print("Order conv rel Linf error standard : ", order_Linf_standard)
print("Order conv rel H1 error standard : ", order_H1int_standard)

print("Order conv rel L2 error diff SW : ", order_L2_diff_SW)
print("Order conv rel Linf error diff SW : ", order_Linf_diff_SW)
print("Order conv rel H1 error diff SW : ", order_H1int_diff_SW)

print("Order conv rel L2 error diff : ", order_L2_diff)
print("Order conv rel Linf error diff : ", order_Linf_diff)
print("Order conv rel H1 error diff : ", order_H1int_diff)

print("Order conv rel L2 error diff 2 : ", order_L2_diff2)
print("Order conv rel Linf error diff 2 : ", order_Linf_diff2)
print("Order conv rel H1 error diff 2 : ", order_H1int_diff2)

#  Write the output file for latex
if conditioning == False:
    f = open("output_no_cond_case1_phiDF.txt", "w")
if conditioning == True:
    f = open("output_cond_case1_phiDF.txt", "w")

f.write("relative L2 norm for phifem, FEM standard, SW, phi-FD, phi-FD2: \n")
f.write("\\addplot[mark=*, blue] coordinates {\n")
output_latex(f, size_mesh_phi_vec, error_L2_phifem_vec)
f.write("};\n\\addplot[mark=*] coordinates {\n")
output_latex(f, size_mesh_standard_vec, error_L2_standard_vec)
f.write("};\n\\addplot[mark=*,green] coordinates {\n")
output_latex(f, size_mesh_diff_SW_vec, error_L2_diff_SW_vec)
f.write("};\n\\addplot[mark=*,red] coordinates {\n")
output_latex(f, size_mesh_diff_vec, error_L2_diff_vec)
f.write("};\n\\addplot[mark=*,orange] coordinates {\n")
output_latex(f, size_mesh_diff2_vec, error_L2_diff2_vec)
f.write("};\n\n")


f.write("relative L infty norm for phifem, FEM standard, SW, phi-FD, phi-FD2: \n")
f.write("\\addplot[mark=*, blue] coordinates {\n")
output_latex(f, size_mesh_phi_vec, error_Linf_phifem_vec)
f.write("};\n\\addplot[mark=*] coordinates {\n")
output_latex(f, size_mesh_standard_vec, error_Linf_standard_vec)
f.write("};\n\\addplot[mark=*,green] coordinates {\n")
output_latex(f, size_mesh_diff_SW_vec, error_Linf_diff_SW_vec)
f.write("};\n\\addplot[mark=*,red] coordinates {\n")
output_latex(f, size_mesh_diff_vec, error_Linf_diff_vec)
f.write("};\n\\addplot[mark=*,orange] coordinates {\n")
output_latex(f, size_mesh_diff2_vec, error_Linf_diff2_vec)
f.write("};\n\n")


f.write(
    "relative H1 norm in Omega_h^Gamma for phifem, FEM standard, SW, phi-FD, phi-FD2: \n"
)
f.write("\\addplot[mark=*, blue] coordinates {\n")
output_latex(f, size_mesh_phi_vec, error_H1int_phifem_vec)
f.write("};\n\\addplot[mark=*] coordinates {\n")
output_latex(f, size_mesh_standard_vec, error_H1int_standard_vec)
f.write("};\n\\addplot[mark=*,green] coordinates {\n")
output_latex(f, size_mesh_diff_SW_vec, error_H1int_diff_SW_vec)
f.write("};\n\\addplot[mark=*,red] coordinates {\n")
output_latex(f, size_mesh_diff_vec, error_H1int_diff_vec)
f.write("};\n\\addplot[mark=*,orange] coordinates {\n")
output_latex(f, size_mesh_diff2_vec, error_H1int_diff2_vec)
f.write("};\n\n")


f.write(
    "relative L2 norm and time standard for phifem, FEM standard, SW, phi-FD, phi-FD2: \n"
)
f.write("\\addplot[mark=*, blue] coordinates {\n")
output_latex(f, error_L2_phifem_vec, time_phifem_vec)
f.write("};\n\\addplot[mark=*] coordinates {\n")
output_latex(f, error_L2_standard_vec, time_standard_vec)
f.write("};\n\\addplot[mark=*,green] coordinates {\n")
output_latex(f, error_L2_diff_SW_vec, time_diff_SW_vec)
f.write("};\n\\addplot[mark=*,red] coordinates {\n")
output_latex(f, error_L2_diff_vec, time_diff_vec)
f.write("};\n\\addplot[mark=*,orange] coordinates {\n")
output_latex(f, error_L2_diff2_vec, time_diff2_vec)
f.write("};\n\n")

f.write("relative Linf norm and time for phifem, FEM standard, SW, phi-FD, phi-FD2: \n")
f.write("\\addplot[mark=*, blue] coordinates {\n")
output_latex(f, error_Linf_phifem_vec, time_phifem_vec)
f.write("};\n\\addplot[mark=*] coordinates {\n")
output_latex(f, error_Linf_standard_vec, time_standard_vec)
f.write("};\n\\addplot[mark=*,green] coordinates {\n")
output_latex(f, error_Linf_diff_SW_vec, time_diff_SW_vec)
f.write("};\n\\addplot[mark=*,red] coordinates {\n")
output_latex(f, error_Linf_diff_vec, time_diff_vec)
f.write("};\n\\addplot[mark=*,orange] coordinates {\n")
output_latex(f, error_Linf_diff2_vec, time_diff2_vec)
f.write("};\n\n")

f.write(
    "relative H1 norm in Omega_h^Gamma and time for phifem, FEM standard, SW, phi-FD, phi-FD2: \n"
)
f.write("\\addplot[mark=*, blue] coordinates {\n")
output_latex(f, error_H1int_phifem_vec, time_phifem_vec)
f.write("};\n\\addplot[mark=*] coordinates {\n")
output_latex(f, error_H1int_standard_vec, time_standard_vec)
f.write("};\n\\addplot[mark=*,green] coordinates {\n")
output_latex(f, error_H1int_diff_SW_vec, time_diff_SW_vec)
f.write("};\n\\addplot[mark=*,red] coordinates {\n")
output_latex(f, error_H1int_diff_vec, time_diff_vec)
f.write("};\n\\addplot[mark=*,orange] coordinates {\n")
output_latex(f, error_H1int_diff2_vec, time_diff2_vec)
f.write("};\n\n")

if conditioning == True:
    f.write("conditioning number for phifem, FEM standard, SW, phi-FD, phi-FD2: \n")
    f.write("\\addplot[mark=*, blue] coordinates {\n")
    output_latex(f, size_mesh_phi_vec, cond_phifem_vec)
    f.write("};\n\\addplot[mark=*] coordinates {\n")
    output_latex(f, size_mesh_standard_vec, cond_standard_vec)
    f.write("};\n\\addplot[mark=*,green] coordinates {\n")
    output_latex(f, size_mesh_diff_SW_vec, cond_diff_SW_vec)
    f.write("};\n\\addplot[mark=*,red] coordinates {\n")
    output_latex(f, size_mesh_diff_vec, cond_diff_vec)
    f.write("};\n\\addplot[mark=*,orange] coordinates {\n")
    output_latex(f, size_mesh_diff2_vec, cond_diff2_vec)
    f.write("};\n\n")

f.write("Order of convergence in L2\n")
f.write(str(round(order_L2_phifem, 2)))
f.write("&")
f.write(str(round(order_L2_standard, 2)))
f.write("&")
f.write(str(round(order_L2_diff_SW, 2)))
f.write("&")
f.write(str(round(order_L2_diff, 2)))
f.write("&")
f.write(str(round(order_L2_diff2, 2)))
f.write("\\\\\n\n")

f.write("Order of convergence in Linf\n")
f.write(str(round(order_Linf_phifem, 2)))
f.write("&")
f.write(str(round(order_Linf_standard, 2)))
f.write("&")
f.write(str(round(order_Linf_diff_SW, 2)))
f.write("&")
f.write(str(round(order_Linf_diff, 2)))
f.write("&")
f.write(str(round(order_Linf_diff2, 2)))
f.write("\\\\\n\n")

f.write("Order of convergence in H1\n")
f.write(str(round(order_H1int_phifem, 2)))
f.write("&")
f.write(str(round(order_H1int_standard, 2)))
f.write("&")
f.write(str(round(order_H1int_diff_SW, 2)))
f.write("&")
f.write(str(round(order_H1int_diff, 2)))
f.write("&")
f.write(str(round(order_H1int_diff2, 2)))
f.write("\\\\\n\n")

f.close()

plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.loglog(size_mesh_standard_vec, error_L2_standard_vec, "-+", label="L2 std")
plt.loglog(size_mesh_phi_vec, error_L2_phifem_vec, "-+", label="L2 phiFEM")
plt.loglog(size_mesh_phi_vec, error_L2_diff_vec, "-+", label="L2 phiFD")
plt.loglog(size_mesh_phi_vec, error_L2_diff_SW_vec, "-+", label="L2 SW")
plt.loglog(size_mesh_phi_vec, error_L2_diff2_vec, "-+", label="L2 phiFD2")

plt.loglog(
    size_mesh_phi_vec,
    [h**2 for h in size_mesh_phi_vec],
    "--",
    label=r"$\mathcal{O}(h)$",
)
plt.legend(ncols=2)

plt.subplot(1, 3, 2)
plt.loglog(size_mesh_standard_vec, error_Linf_standard_vec, "-+", label="Linf std")
plt.loglog(size_mesh_phi_vec, error_Linf_phifem_vec, "-+", label="Linf phiFEM")
plt.loglog(size_mesh_phi_vec, error_Linf_diff_vec, "-+", label="Linf phiFD")
plt.loglog(size_mesh_phi_vec, error_Linf_diff_SW_vec, "-+", label="Linf SW")
plt.loglog(size_mesh_phi_vec, error_Linf_diff2_vec, "-+", label="Linf phiFD2")
plt.loglog(
    size_mesh_phi_vec, [h for h in size_mesh_phi_vec], "--", label=r"$\mathcal{O}(h)$"
)
plt.loglog(
    size_mesh_phi_vec,
    [h**2 for h in size_mesh_phi_vec],
    "--",
    label=r"$\mathcal{O}(h)$",
)
plt.legend(ncols=2)

plt.subplot(1, 3, 3)
plt.loglog(size_mesh_standard_vec, error_H1int_standard_vec, "-+", label="H1 std")
plt.loglog(size_mesh_phi_vec, error_H1int_phifem_vec, "-+", label="H1 phiFEM")
plt.loglog(size_mesh_phi_vec, error_H1int_diff_vec, "-+", label="H1 phiFD")
plt.loglog(size_mesh_phi_vec, error_H1int_diff_SW_vec, "-+", label="H1 SW")
plt.loglog(size_mesh_phi_vec, error_H1int_diff2_vec, "-+", label="H1 phiFD2")
plt.loglog(
    size_mesh_phi_vec, [h for h in size_mesh_phi_vec], "--", label=r"$\mathcal{O}(h)$"
)
plt.loglog(
    size_mesh_phi_vec,
    [h**2 for h in size_mesh_phi_vec],
    "--",
    label=r"$\mathcal{O}(h)$",
)
plt.legend(ncols=2)

plt.savefig("errors_3.pdf")
plt.show()
