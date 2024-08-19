import numpy as np
import dolfin as df
import sympy
import matplotlib.pyplot as plt
from time import time
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import mshr
import os
import seaborn as sns

sns.set_theme("paper")

df.parameters["ghost_mode"] = "shared_facet"
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True
df.parameters["allow_extrapolation"] = True
df.parameters["form_compiler"]["representation"] = "uflacs"

# radius of the domains
R_x = 0.3  # +1e-10
R_y = 0.3  # +1e-10
R_z = 0.3

# Polynome Pk
polV = 1
polPhi = polV + 1
# parameters["form_compiler"]["quadrature_degree"]=2*(polV+polPhi)

# Ghost penalty
ghost = True

# plot the solution
Plot = False

# Compute the conditioning number
conditioning = False


def Omega(x, y, z):
    return (
        -1.0 + ((x - 0.5) / R_x) ** 2 + ((y - 0.5) / R_y) ** 2 + ((z - 0.5) / R_z) ** 2
        <= 3e-16
    )


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
x_symb, y_symb, z_symb = sympy.symbols("xx yy zz")
r = sympy.sqrt(
    ((x_symb - 0.5) / R_x) ** 2
    + ((y_symb - 0.5) / R_y) ** 2
    + ((z_symb - 0.5) / R_z) ** 2
    + 1e-12
)
K_R = sympy.pi / 2.0
u1 = sympy.cos(K_R * r)
f1 = -(
    sympy.diff(sympy.diff(u1, x_symb), x_symb)
    + sympy.diff(sympy.diff(u1, y_symb), y_symb)
    + sympy.diff(sympy.diff(u1, z_symb), z_symb)
)
g1 = 0.0

###########################################
### beginning phi fem ###########
###########################################

# parameter of the ghost penalty
sigma = 1.0

sizes = [12, 16, 20, 24, 28, 32, 40]
sizes_std = [8, 12, 16, 20, 24, 28, 32]

Iter = len(sizes)


if os.path.exists("./time_phifem_vec.npy"):
    size_mesh_phi_vec = np.zeros(Iter)
    for iii in range(Iter):
        print("#########################")
        print("## Iteration phifem ", iii + 1, "##")
        print("#########################")

        # Construction of the mesh
        N = sizes[iii]
        print("N=", N)
        mesh_macro = df.BoxMesh(
            df.Point(0.0, 0.0, 0.0), df.Point(1.0, 1.0, 1.0), N + 1, N + 1, N + 1
        )
        size_mesh_phi_vec[iii] = mesh_macro.hmax()
else:
    size_mesh_phi_vec = np.zeros(Iter)
    error_L2_phifem_vec = np.zeros(Iter)
    error_Linf_phifem_vec = np.zeros(Iter)
    error_H1_phifem_vec = np.zeros(Iter)
    error_H1int_phifem_vec = np.zeros(Iter)
    cond_phifem_vec = np.zeros(Iter)
    time_phifem_vec = np.zeros(Iter)
    for iii in range(Iter):
        print("#########################")
        print("## Iteration phifem ", iii + 1, "##")
        print("#########################")

        t_init = time()
        # Construction of the mesh
        N = sizes[iii]
        print("N=", N)
        mesh_macro = df.BoxMesh(
            df.Point(0.0, 0.0, 0.0), df.Point(1.0, 1.0, 1.0), N, N, N
        )
        size_mesh_phi_vec[iii] = mesh_macro.hmax()
        domains = df.MeshFunction("size_t", mesh_macro, mesh_macro.topology().dim())
        domains.set_all(0)
        for ind in range(mesh_macro.num_cells()):
            mycell = df.Cell(mesh_macro, ind)
            v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, v4x, v4y, v4z = (
                mycell.get_vertex_coordinates()
            )
            if (
                Omega(v1x, v1y, v1z)
                or Omega(v2x, v2y, v2z)
                or Omega(v3x, v3y, v3z)
                or Omega(v4x, v4y, v4z)
            ):
                domains[ind] = 1
        time_macro_mesh = time() - t_init
        print("time macro", time_macro_mesh)
        t_init = time()
        mesh = df.SubMesh(mesh_macro, domains, 1)
        print("num cells:", mesh.num_cells())
        V = df.FunctionSpace(mesh, "CG", polV)
        V_phi = df.FunctionSpace(mesh, "CG", polPhi)
        time_mesh_space = time() - t_init
        print("time mesh and space", time_mesh_space)

        t_init = time()
        # Construction of phi
        phi = df.Expression(
            "-1.0 + pow((x[0]-0.5) / R_x,2) + pow((x[1]-0.5) / R_y,2) + pow((x[2]-0.5) / R_z,2)",
            R_x=R_x,
            R_y=R_y,
            R_z=R_z,
            degree=polPhi,
            domain=mesh,
        )
        phi = df.interpolate(phi, V_phi)

        # Computation of the source term
        f_expr = df.Expression(
            sympy.ccode(f1)
            .replace("xx", "x[0]")
            .replace("yy", "x[1]")
            .replace("zz", "x[2]"),
            degree=polV,
            domain=mesh,
        )
        u_expr = df.Expression(
            sympy.ccode(u1)
            .replace("xx", "x[0]")
            .replace("yy", "x[1]")
            .replace("zz", "x[2]"),
            degree=4,
            domain=mesh,
        )
        g = df.Expression(
            sympy.ccode(g1)
            .replace("xx", "x[0]")
            .replace("yy", "x[1]")
            .replace("zz", "x[2]"),
            degree=polV + 2,
            domain=mesh,
        )

        # Facets and cells where we apply the ghost penalty
        mesh.init(1, 2)
        facet_ghost = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        cell_ghost = df.MeshFunction("size_t", mesh, mesh.topology().dim())
        facet_ghost.set_all(0)
        cell_ghost.set_all(0)
        count_cell_ghost = 0
        for mycell in df.cells(mesh):
            for myfacet in df.facets(mycell):
                v1, v2, v3 = df.vertices(myfacet)
                if (
                    (
                        phi(v1.point().x(), v1.point().y(), v1.point().z())
                        * phi(v2.point().x(), v2.point().y(), v2.point().z())
                        <= df.DOLFIN_EPS
                    )
                    or (
                        phi(v1.point().x(), v1.point().y(), v1.point().z())
                        * phi(v3.point().x(), v3.point().y(), v3.point().z())
                        <= df.DOLFIN_EPS
                    )
                    or (
                        phi(v2.point().x(), v2.point().y(), v2.point().z())
                        * phi(v3.point().x(), v3.point().y(), v3.point().z())
                        <= df.DOLFIN_EPS
                    )
                ):
                    cell_ghost[mycell] = 1
                    for myfacet2 in df.facets(mycell):
                        facet_ghost[myfacet2] = 1

        for mycell in df.cells(mesh):
            if cell_ghost[mycell] == 1:
                count_cell_ghost += 1
        print("num of cell in the ghost penalty:", count_cell_ghost)
        time_select_cell = time() - t_init
        print("time select cell", time_select_cell)
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
        if ghost == False:
            a = (
                df.inner(df.grad(phi * u), df.grad(phi * v)) * dx
                - df.dot(df.inner(df.grad(phi * u), n), phi * v) * ds
            )
            L = (
                f_expr * v * phi * dx
                + df.inner(df.grad(-g), df.grad(phi * v)) * dx
                - df.dot(df.inner(df.grad(-g), n), phi * v) * ds
            )
        if ghost == True:
            a = (
                df.inner(df.grad(phi * u), df.grad(phi * v)) * dx
                - df.dot(df.inner(df.grad(phi * u), n), phi * v) * ds
                + sigma
                * df.avg(h)
                * df.dot(df.jump(df.grad(phi * u), n), df.jump(df.grad(phi * v), n))
                * dS(1)
                + sigma
                * h**2
                * df.inner(
                    phi * u + df.div(df.grad(phi * u)),
                    phi * v + df.div(df.grad(phi * v)),
                )
                * dx(1)
            )
            L = (
                f_expr * v * phi * dx
                - sigma
                * h**2
                * df.inner(f_expr, phi * v + df.div(df.grad(phi * v)))
                * dx(1)
                + df.inner(df.grad(-g), df.grad(phi * v)) * dx
                - df.dot(df.inner(df.grad(-g), n), phi * v) * ds
                + sigma
                * df.avg(h)
                * df.dot(df.jump(df.grad(-g), n), df.jump(df.grad(phi * v), n))
                * dS(1)
                + sigma
                * h**2
                * df.inner(df.div(df.grad(-g)), phi * v + df.div(df.grad(phi * v)))
                * dx(1)
            )

        # Define solution function
        u_h = df.Function(V)
        print("ready to solve")
        df.solve(
            a == L,
            u_h,
            solver_parameters={"linear_solver": "gmres", "preconditioner": "hypre_amg"},
        )

        sol = u_h * phi + g
        t_final = time()
        sol = df.project(sol, V)

        # computation of the error
        V_macro = df.FunctionSpace(mesh_macro, "CG", 1)
        sol2 = df.interpolate(sol, V_macro)
        Nx = N
        Ny = N
        Nz = N
        x = np.linspace(0.0, 1.0, Nx + 1)
        y = np.linspace(0.0, 1.0, Ny + 1)
        z = np.linspace(0.0, 1.0, Nz + 1)

        phi_np = (
            lambda x, y, z: -1.0
            + ((x - 0.5) / R_x) ** 2
            + ((y - 0.5) / R_y) ** 2
            + ((z - 0.5) / R_z) ** 2
        )

        Ndof = (Nx + 1) * (Ny + 1) * (Nz + 1)
        hx = x[1] - x[0]
        hy = y[1] - y[0]
        hz = z[1] - z[0]
        X, Y, Z = np.meshgrid(x, y, z)  # 2D meshgrid

        sol_values = np.zeros((Ny + 1, Nx + 1, Nz + 1))
        for ix in range(Nx + 1):
            for iy in range(Ny + 1):
                for iz in range(Nz + 1):
                    sol_values[iy, ix, iz] = sol2(x[ix], y[iy], z[iz])

        phijik = phi_np(X, Y, Z)
        ind = (phijik < 0) + 0
        indOut = 1 - ind
        ue = sympy.lambdify([x_symb, y_symb, z_symb], u1)
        uref = ue(X, Y, Z)

        e = sol_values - uref

        eL2 = np.sqrt(np.sum(e * e * (1 - indOut))) / np.sqrt(
            np.sum(uref * uref * (1 - indOut))
        )
        emax = np.max(np.abs(e * (1 - indOut))) / np.max(np.abs(uref * (1 - indOut)))

        ex = (e[:, 1 : Nx + 1, :] - e[:, 0:Nx, :]) / hx
        urefx = (uref[:, 1 : Nx + 1, :] - uref[:, 0:Nx, :]) / hx
        intdx = (ind[:, 1 : Nx + 1, :] + ind[:, 0:Nx, :] == 2) + 0
        fulldx = (ind[:, 1 : Nx + 1, :] + ind[:, 0:Nx, :] > 0) + 0

        ey = (e[1 : Ny + 1, :, :] - e[0:Ny, :, :]) / hy
        urefy = (uref[1 : Ny + 1, :, :] - uref[0:Ny, :, :]) / hy
        intdy = (ind[1 : Ny + 1, :, :] + ind[0:Ny, :, :] == 2) + 0
        fulldy = (ind[1 : Ny + 1, :, :] + ind[0:Ny, :, :] > 0) + 0

        ez = (e[:, :, 1 : Nz + 1] - e[:, :, 0:Nz]) / hz
        urefz = (uref[:, :, 1 : Nz + 1] - uref[:, :, 0:Nz]) / hz
        intdz = (ind[:, :, 1 : Nz + 1] + ind[:, :, 0:Nz] == 2) + 0
        fulldz = (ind[:, :, 1 : Nz + 1] + ind[:, :, 0:Nz] > 0) + 0
        eH1 = np.sqrt(
            (
                np.sum(ex * ex * fulldx)
                + np.sum(ey * ey * fulldy)
                + np.sum(ez * ez * fulldz)
            )
        ) / np.sqrt(
            (
                np.sum(urefx * urefx * fulldx)
                + np.sum(urefy * urefy * fulldy)
                + np.sum(urefz * urefz * fulldz)
            )
        )
        eH1int = np.sqrt(
            (
                np.sum(ex * ex * intdx)
                + np.sum(ey * ey * intdy)
                + np.sum(ez * ez * intdz)
            )
        ) / np.sqrt(
            (
                np.sum(urefx * urefx * intdx)
                + np.sum(urefy * urefy * intdy)
                + np.sum(urefz * urefz * intdz)
            )
        )

        print("h :", mesh.hmax())
        print("relative L2 error phiFEM : ", eL2)
        print("relative L inf error phiFEM : ", emax)
        print("relative H1 int error phiFEM : ", eH1int)
        print("time : ", t_final - t_init)
        if conditioning == True:
            A = np.matrix(df.assemble(a).array())
            cond = np.linalg.cond(A)
            cond_phifem_vec[iii] = cond
            print("conditioning number phiFEM : ", cond)
        time_phifem_vec[iii] = t_final - t_init
        error_L2_phifem_vec[iii] = eL2
        error_Linf_phifem_vec[iii] = emax
        error_H1_phifem_vec[iii] = eH1
        error_H1int_phifem_vec[iii] = eH1int
        size_mesh_phi_vec[iii] = mesh.hmax()
    np.save("time_phifem_vec.npy", time_phifem_vec)
    np.save("error_L2_phifem_vec.npy", error_L2_phifem_vec)
    np.save("error_Linf_phifem_vec.npy", error_Linf_phifem_vec)
    np.save("error_H1int_phifem_vec.npy", error_H1int_phifem_vec)


###########################################
######### beginning standard fem ##########
###########################################

# Initialistion of the output
size_mesh_standard_vec = np.zeros(Iter)
error_L2_standard_vec = np.zeros(Iter)
error_Linf_standard_vec = np.zeros(Iter)
error_H1_standard_vec = np.zeros(Iter)
error_H1int_standard_vec = np.zeros(Iter)
cond_standard_vec = np.zeros(Iter)
time_standard_vec = np.zeros(Iter)

domain = mshr.Ellipsoid(
    df.Point(0.5, 0.5, 0.5), R_x, R_y, R_z
)  # creation of the domain
if os.path.exists("./time_standard_vec.npy"):
    for i in range(Iter):
        print("#########################")
        print("## Iteration standard ", i + 1, "##")
        print("#########################")
        # Construction of the mesh
        H = sizes_std[i]
        mesh = mshr.generate_mesh(domain, H)
        size_mesh_standard_vec[i] = mesh.hmax()

else:
    for i in range(Iter):
        print("#########################")
        print("## Iteration standard ", i + 1, "##")
        print("#########################")
        # Construction of the mesh

        t_init = time()
        # Construction of the mesh
        H = sizes_std[i]
        mesh = mshr.generate_mesh(domain, H)
        print("num cells:", mesh.num_cells())
        V = df.FunctionSpace(mesh, "CG", polV)

        # Computation of the source term
        f_expr = df.Expression(
            sympy.ccode(f1)
            .replace("xx", "x[0]")
            .replace("yy", "x[1]")
            .replace("zz", "x[2]"),
            degree=polV,
            domain=mesh,
        )
        u_expr = df.Expression(
            sympy.ccode(u1)
            .replace("xx", "x[0]")
            .replace("yy", "x[1]")
            .replace("zz", "x[2]"),
            degree=4,
            domain=mesh,
        )
        g = df.Expression(
            sympy.ccode(g1)
            .replace("xx", "x[0]")
            .replace("yy", "x[1]")
            .replace("zz", "x[2]"),
            degree=4,
            domain=mesh,
        )
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
        df.solve(
            a == L,
            sol,
            bc,
            solver_parameters={"linear_solver": "gmres", "preconditioner": "hypre_amg"},
        )
        print("solved")
        t_final = time()

        # Computation of the error
        N = sizes[i]
        print("N=", N)
        mesh_macro = df.BoxMesh(
            df.Point(0.0, 0.0, 0.0), df.Point(1.0, 1.0, 1.0), N, N, N
        )
        V_macro = df.FunctionSpace(mesh_macro, "CG", 1)
        sol2 = df.interpolate(sol, V_macro)
        Nx = N
        Ny = N
        Nz = N
        x = np.linspace(0.0, 1.0, Nx + 1)
        y = np.linspace(0.0, 1.0, Ny + 1)
        z = np.linspace(0.0, 1.0, Nz + 1)

        phi = (
            lambda x, y, z: -1.0
            + ((x - 0.5) / R_x) ** 2
            + ((y - 0.5) / R_y) ** 2
            + ((z - 0.5) / R_z) ** 2
        )

        Ndof = (Nx + 1) * (Ny + 1) * (Nz + 1)
        hx = x[1] - x[0]
        hy = y[1] - y[0]
        hz = z[1] - z[0]
        X, Y, Z = np.meshgrid(x, y, z)

        sol_values = np.zeros((Ny + 1, Nx + 1, Nz + 1))
        for ix in range(Nx + 1):
            for iy in range(Ny + 1):
                for iz in range(Nz + 1):
                    sol_values[iy, ix, iz] = sol2(x[ix], y[iy], z[iz])
        phijik = phi(X, Y, Z)
        ind = (phijik < 0) + 0
        indOut = 1 - ind
        ue = sympy.lambdify([x_symb, y_symb, z_symb], u1)
        uref = ue(X, Y, Z)

        e = sol_values - uref

        eL2 = np.sqrt(np.sum(e * e * (1 - indOut))) / np.sqrt(
            np.sum(uref * uref * (1 - indOut))
        )
        emax = np.max(np.abs(e * (1 - indOut))) / np.max(np.abs(uref * (1 - indOut)))

        ex = (e[:, 1 : Nx + 1, :] - e[:, 0:Nx, :]) / hx
        urefx = (uref[:, 1 : Nx + 1, :] - uref[:, 0:Nx, :]) / hx
        intdx = (ind[:, 1 : Nx + 1, :] + ind[:, 0:Nx, :] == 2) + 0
        fulldx = (ind[:, 1 : Nx + 1, :] + ind[:, 0:Nx, :] > 0) + 0

        ey = (e[1 : Ny + 1, :, :] - e[0:Ny, :, :]) / hy
        urefy = (uref[1 : Ny + 1, :, :] - uref[0:Ny, :, :]) / hy
        intdy = (ind[1 : Ny + 1, :, :] + ind[0:Ny, :, :] == 2) + 0
        fulldy = (ind[1 : Ny + 1, :, :] + ind[0:Ny, :, :] > 0) + 0

        ez = (e[:, :, 1 : Nz + 1] - e[:, :, 0:Nz]) / hz
        urefz = (uref[:, :, 1 : Nz + 1] - uref[:, :, 0:Nz]) / hz
        intdz = (ind[:, :, 1 : Nz + 1] + ind[:, :, 0:Nz] == 2) + 0
        fulldz = (ind[:, :, 1 : Nz + 1] + ind[:, :, 0:Nz] > 0) + 0
        eH1 = np.sqrt(
            (
                np.sum(ex * ex * fulldx)
                + np.sum(ey * ey * fulldy)
                + np.sum(ez * ez * fulldz)
            )
        ) / np.sqrt(
            (
                np.sum(urefx * urefx * fulldx)
                + np.sum(urefy * urefy * fulldy)
                + np.sum(urefz * urefz * fulldz)
            )
        )
        eH1int = np.sqrt(
            (
                np.sum(ex * ex * intdx)
                + np.sum(ey * ey * intdy)
                + np.sum(ez * ez * intdz)
            )
        ) / np.sqrt(
            (
                np.sum(urefx * urefx * intdx)
                + np.sum(urefy * urefy * intdy)
                + np.sum(urefz * urefz * intdz)
            )
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

    np.save("time_standard_vec.npy", time_standard_vec)
    np.save("error_L2_standard_vec.npy", error_L2_standard_vec)
    np.save("error_Linf_standard_vec.npy", error_Linf_standard_vec)
    np.save("error_H1int_standard_vec.npy", error_H1int_standard_vec)

time_standard_vec = np.load("time_standard_vec.npy")
time_phifem_vec = np.load("time_phifem_vec.npy")
error_L2_standard_vec = np.load("error_L2_standard_vec.npy")
error_L2_phifem_vec = np.load("error_L2_phifem_vec.npy")
error_Linf_standard_vec = np.load("error_Linf_standard_vec.npy")
error_Linf_phifem_vec = np.load("error_Linf_phifem_vec.npy")
error_H1int_standard_vec = np.load("error_H1int_standard_vec.npy")
error_H1int_phifem_vec = np.load("error_H1int_phifem_vec.npy")


###########################################
############  beginning PhiFD  ############
############   First  scheme   ############
###########################################
sigma = 0.01
gamma = 10.0

# Initialistion of the output
interp_errors = np.zeros(Iter)
size_mesh_diff_vec = np.zeros(Iter)
error_L2_diff_vec = np.zeros(Iter)
error_Linf_diff_vec = np.zeros(Iter)
error_H1_diff_vec = np.zeros(Iter)
error_H1int_diff_vec = np.zeros(Iter)
cond_diff_vec = np.zeros(Iter)
time_diff_vec = np.zeros(Iter)
for iii in range(Iter):
    print("######################")
    print("## Iteration diff 2 ", iii + 1, "##")
    print("######################")

    # begin time
    t_init = time()

    # Construction of the mesh
    N = sizes[iii]
    print("N=", N)
    Nx = N
    Ny = N
    Nz = N
    x = np.linspace(0.0, 1.0, Nx + 1)
    y = np.linspace(0.0, 1.0, Ny + 1)
    z = np.linspace(0.0, 1.0, Nz + 1)
    ue = sympy.lambdify([x_symb, y_symb, z_symb], u1)
    f = sympy.lambdify([x_symb, y_symb, z_symb], f1)
    phi = (
        lambda x, y, z: -1.0
        + ((x - 0.5) / R_x) ** 2
        + ((y - 0.5) / R_y) ** 2
        + ((z - 0.5) / R_z) ** 2
    )

    Ndof = (Nx + 1) * (Ny + 1) * (Nz + 1)
    hx = x[1] - x[0]
    hy = y[1] - y[0]
    hz = z[1] - z[0]
    X, Y, Z = np.meshgrid(x, y, z)  # 3D meshgrid
    phijik = phi(X, Y, Z)
    ind = (phijik < 0) + 0
    mask = sp.diags(diagonals=ind.ravel())
    indOut = 1 - ind

    # laplacian matrix
    D2x = (1.0 / hx / hx) * sp.diags(
        diagonals=[-1, 2, -1], offsets=[-1, 0, 1], shape=(Nx + 1, Nx + 1)
    )
    D2y = (1.0 / hy / hy) * sp.diags(
        diagonals=[-1, 2, -1], offsets=[-1, 0, 1], shape=(Ny + 1, Ny + 1)
    )
    D2z = (1.0 / hz / hz) * sp.diags(
        diagonals=[-1, 2, -1], offsets=[-1, 0, 1], shape=(Nz + 1, Nz + 1)
    )

    D2x_2d = sp.kron(sp.kron(sp.eye(Ny + 1), D2x), sp.eye(Nz + 1))
    D2y_2d = sp.kron(sp.kron(D2y, sp.eye(Nx + 1)), sp.eye(Nz + 1))
    D2z_2d = sp.kron(sp.kron(sp.eye(Ny + 1), sp.eye(Nx + 1)), D2z)

    A = mask @ (D2x_2d + D2y_2d + D2z_2d)
    row = []
    col = []
    coef = []  # for the matrix implementing BC

    def rav(i, j, k):
        return np.ravel_multi_index([j, i, k], (Ny + 1, Nx + 1, Nz + 1))

    def AddMat(eq, i, j, k, a):
        row.append(eq)
        col.append(rav(i, j, k))
        coef.append(a)

    # active sites for the ghost penalty
    actGx = np.zeros((Ny + 1, Nx + 1, Nz + 1))
    actGy = np.zeros((Ny + 1, Nx + 1, Nz + 1))
    actGz = np.zeros((Ny + 1, Nx + 1, Nz + 1))

    indx = ind[:, 1 : Nx + 1, :] - ind[:, 0:Nx, :]
    J, I, K = np.where((indx == 1) | (indx == -1))
    for k in range(np.shape(I)[0]):
        if indx[J[k], I[k], K[k]] == 1:
            indOut[J[k], I[k], K[k]] = 0
            actGx[J[k], I[k] + 1, K[k]] = 1
        else:
            indOut[J[k], I[k] + 1, K[k]] = 0
            actGx[J[k], I[k], K[k]] = 1

        i, j, k_ = I[k], J[k], K[k]
        phiS = np.square(phijik[j, i, k_]) + np.square(phijik[j, i + 1, k_])
        phii = phijik[j, i, k_]
        phiip1 = phijik[j, i + 1, k_]
        AddMat(rav(i, j, k_), i, j, k_, phiip1 * phiip1 / phiS)
        AddMat(rav(i, j, k_), i + 1, j, k_, -phii * phiip1 / phiS)
        AddMat(rav(i + 1, j, k_), i, j, k_, -phii * phiip1 / phiS)
        AddMat(rav(i + 1, j, k_), i + 1, j, k_, phii * phii / phiS)

    indy = ind[1 : Ny + 1, :, :] - ind[0:Ny, :, :]
    J, I, K = np.where((indy == 1) | (indy == -1))
    for k in range(np.shape(I)[0]):
        if indy[J[k], I[k], K[k]] == 1:
            indOut[J[k], I[k], K[k]] = 0
            actGy[J[k] + 1, I[k], K[k]] = 1
        else:
            indOut[J[k] + 1, I[k], K[k]] = 0
            actGy[J[k], I[k], K[k]] = 1

        i, j, k_ = I[k], J[k], K[k]
        phiS = np.square(phijik[j, i, k_]) + np.square(phijik[j + 1, i, k_])
        phij = phijik[j, i, k_]
        phijp1 = phijik[j + 1, i, k_]
        AddMat(rav(i, j, k_), i, j, k_, phijp1 * phijp1 / phiS)
        AddMat(rav(i, j, k_), i, j + 1, k_, -phij * phijp1 / phiS)
        AddMat(rav(i, j + 1, k_), i, j, k_, -phij * phijp1 / phiS)
        AddMat(rav(i, j + 1, k_), i, j + 1, k_, phij * phij / phiS)

    indz = ind[:, :, 1 : Nz + 1] - ind[:, :, 0:Nz]
    J, I, K = np.where((indz == 1) | (indz == -1))
    for k in range(np.shape(I)[0]):
        if indz[J[k], I[k], K[k]] == 1:
            indOut[J[k], I[k], K[k]] = 0
            actGz[J[k], I[k], K[k] + 1] = 1
        else:
            indOut[J[k], I[k], K[k] + 1] = 0
            actGz[J[k], I[k], K[k]] = 1

        i, j, k_ = I[k], J[k], K[k]
        phiS = np.square(phijik[j, i, k_]) + np.square(phijik[j, i, k_ + 1])
        phik = phijik[j, i, k_]
        phikp1 = phijik[j, i, k_ + 1]
        AddMat(rav(i, j, k_), i, j, k_, phikp1 * phikp1 / phiS)
        AddMat(rav(i, j, k_), i, j, k_ + 1, -phik * phikp1 / phiS)
        AddMat(rav(i, j, k_ + 1), i, j, k_, -phik * phikp1 / phiS)
        AddMat(rav(i, j, k_ + 1), i, j, k_ + 1, phik * phik / phiS)

    npcoef = (gamma / hx / hy) * np.array(coef)
    B = sp.coo_array((npcoef, (row, col)), shape=(Ndof, Ndof))

    # ghost penalty
    maskGx = sp.diags(diagonals=actGx.ravel())
    maskGy = sp.diags(diagonals=actGy.ravel())
    maskGz = sp.diags(diagonals=actGz.ravel())

    C = sigma * (
        hx**2 * (D2x_2d.T @ maskGx @ D2x_2d)
        + hy**2 * (D2y_2d.T @ maskGy @ D2y_2d)
        + hz**2 * (D2z_2d.T @ maskGz @ D2z_2d)
    )
    # penalization outside
    D = sp.diags(diagonals=indOut.ravel())

    # linear system
    A = (A + B + C + D).tocsr()
    b = f(X, Y, Z)
    b = (ind * b).ravel()
    u = spsolve(A, b).reshape(Ny + 1, Nx + 1, Nz + 1)
    # final time
    t_final = time()
    uref = ue(X, Y, Z)
    e = u - uref

    eL2 = np.sqrt(np.sum(e * e * ind)) / np.sqrt(np.sum(uref * uref * ind))
    emax = np.max(np.abs(e * ind)) / np.max(np.abs(uref * ind))
    ex = (e[:, 1 : Nx + 1, :] - e[:, 0:Nx, :]) / hx
    urefx = (uref[:, 1 : Nx + 1, :] - uref[:, 0:Nx, :]) / hx
    intdx = (ind[:, 1 : Nx + 1, :] + ind[:, 0:Nx, :] == 2) + 0
    fulldx = (ind[:, 1 : Nx + 1, :] + ind[:, 0:Nx, :] > 0) + 0

    ey = (e[1 : Ny + 1, :, :] - e[0:Ny, :, :]) / hy
    urefy = (uref[1 : Ny + 1, :, :] - uref[0:Ny, :, :]) / hy
    intdy = (ind[1 : Ny + 1, :, :] + ind[0:Ny, :, :] == 2) + 0
    fulldy = (ind[1 : Ny + 1, :, :] + ind[0:Ny, :, :] > 0) + 0

    ez = (e[:, :, 1 : Nz + 1] - e[:, :, 0:Nz]) / hz
    urefz = (uref[:, :, 1 : Nz + 1] - uref[:, :, 0:Nz]) / hz
    intdz = (ind[:, :, 1 : Nz + 1] + ind[:, :, 0:Nz] == 2) + 0
    fulldz = (ind[:, :, 1 : Nz + 1] + ind[:, :, 0:Nz] > 0) + 0

    eH1 = np.sqrt(
        (np.sum(ex * ex * fulldx) + np.sum(ey * ey * fulldy) + np.sum(ez * ez * fulldz))
    ) / np.sqrt(
        (
            np.sum(urefx * urefx * fulldx)
            + np.sum(urefy * urefy * fulldy)
            + np.sum(urefz * urefz * fulldz)
        )
    )
    eH1int = np.sqrt(
        (np.sum(ex * ex * intdx) + np.sum(ey * ey * intdy) + np.sum(ez * ez * intdz))
    ) / np.sqrt(
        (
            np.sum(urefx * urefx * intdx)
            + np.sum(urefy * urefy * intdy)
            + np.sum(urefz * urefz * intdz)
        )
    )

    print("h :", np.sqrt(2) / N)
    print("relative L2 error diff : ", eL2)
    print("relative L inf error diff : ", emax)
    print("relative H1 error diff : ", eH1)
    print("relative H1 error diff int : ", eH1int)
    print("time : ", t_final - t_init)
    if conditioning == True:
        cond = np.linalg.cond(A.todense())
        cond_diff_vec[iii] = cond
        print("conditioning number diff : ", cond)
    time_diff_vec[iii] = t_final - t_init
    error_L2_diff_vec[iii] = eL2
    error_Linf_diff_vec[iii] = emax
    error_H1_diff_vec[iii] = eH1
    error_H1int_diff_vec[iii] = eH1int
    size_mesh_diff_vec[iii] = np.sqrt(2) / N


###########################################
############  beginning PhiFD  ############
############   Second scheme   ############
###########################################
sigma = 0.0
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
for iii in range(Iter):
    print("######################")
    print("## Iteration diff 2 ", iii + 1, "##")
    print("######################")

    # begin time
    t_init = time()
    N = sizes[iii]
    print("N=", N)
    Nx = N
    Ny = N
    Nz = N
    x = np.linspace(0.0, 1.0, Nx + 1)
    y = np.linspace(0.0, 1.0, Ny + 1)
    z = np.linspace(0.0, 1.0, Nz + 1)
    ue = sympy.lambdify([x_symb, y_symb, z_symb], u1)
    f = sympy.lambdify([x_symb, y_symb, z_symb], f1)
    phi = (
        lambda x, y, z: -1.0
        + ((x - 0.5) / R_x) ** 2
        + ((y - 0.5) / R_y) ** 2
        + ((z - 0.5) / R_z) ** 2
    )

    Ndof = (Nx + 1) * (Ny + 1) * (Nz + 1)
    hx = x[1] - x[0]
    hy = y[1] - y[0]
    hz = z[1] - z[0]
    X, Y, Z = np.meshgrid(x, y, z)  # 3D meshgrid
    phijik = phi(X, Y, Z)
    ind = (phijik < 0) + 0
    mask = sp.diags(diagonals=ind.ravel())
    indOut = 1 - ind

    # laplacian matrix
    D2x = (1.0 / hx / hx) * sp.diags(
        diagonals=[-1, 2, -1], offsets=[-1, 0, 1], shape=(Nx + 1, Nx + 1)
    )
    D2y = (1.0 / hy / hy) * sp.diags(
        diagonals=[-1, 2, -1], offsets=[-1, 0, 1], shape=(Ny + 1, Ny + 1)
    )
    D2z = (1.0 / hz / hz) * sp.diags(
        diagonals=[-1, 2, -1], offsets=[-1, 0, 1], shape=(Nz + 1, Nz + 1)
    )

    D2x_2d = sp.kron(sp.kron(sp.eye(Ny + 1), D2x), sp.eye(Nz + 1))
    D2y_2d = sp.kron(sp.kron(D2y, sp.eye(Nx + 1)), sp.eye(Nz + 1))
    D2z_2d = sp.kron(sp.kron(sp.eye(Ny + 1), sp.eye(Nx + 1)), D2z)

    A = mask @ (D2x_2d + D2y_2d + D2z_2d)
    row = []
    col = []
    coef = []

    def rav(i, j, k):
        return np.ravel_multi_index([j, i, k], (Ny + 1, Nx + 1, Nz + 1))

    def AddMat(eq, i, j, k, a):
        row.append(eq)
        col.append(rav(i, j, k))
        coef.append(a)

    # active sites for the ghost penalty
    actGx = np.zeros((Ny + 1, Nx + 1, Nz + 1))
    actGy = np.zeros((Ny + 1, Nx + 1, Nz + 1))
    actGz = np.zeros((Ny + 1, Nx + 1, Nz + 1))

    indx = ind[:, 1 : Nx + 1, :] - ind[:, 0:Nx, :]
    J, I, K = np.where((indx == 1) | (indx == -1))
    for k in range(np.shape(I)[0]):
        if indx[J[k], I[k], K[k]] == 1:
            i, j, k_ = I[k] + 1, J[k], K[k]
            indOut[J[k], I[k], K[k]] = 0
        else:
            i, j, k_ = I[k], J[k], K[k]
            indOut[J[k], I[k] + 1, K[k]] = 0
        ai = 2 * phijik[j, i + 1, k_] * phijik[j, i - 1, k_]
        aim1 = -phijik[j, i, k_] * phijik[j, i + 1, k_]
        aip1 = -phijik[j, i, k_] * phijik[j, i - 1, k_]
        phiS = ai**2 + aim1**2 + aip1**2
        AddMat(rav(i - 1, j, k_), i - 1, j, k_, aim1 * aim1 / phiS)
        AddMat(rav(i - 1, j, k_), i, j, k_, aim1 * ai / phiS)
        AddMat(rav(i - 1, j, k_), i + 1, j, k_, aim1 * aip1 / phiS)
        AddMat(rav(i, j, k_), i - 1, j, k_, ai * aim1 / phiS)
        AddMat(rav(i, j, k_), i, j, k_, ai * ai / phiS)
        AddMat(rav(i, j, k_), i + 1, j, k_, ai * aip1 / phiS)
        AddMat(rav(i + 1, j, k_), i - 1, j, k_, aip1 * aim1 / phiS)
        AddMat(rav(i + 1, j, k_), i, j, k_, aip1 * ai / phiS)
        AddMat(rav(i + 1, j, k_), i + 1, j, k_, aip1 * aip1 / phiS)

    indy = ind[1 : Ny + 1, :, :] - ind[0:Ny, :, :]
    J, I, K = np.where((indy == 1) | (indy == -1))
    for k in range(np.shape(I)[0]):
        if indy[J[k], I[k], K[k]] == 1:
            i, j, k_ = I[k], J[k] + 1, K[k]
            indOut[J[k], I[k], K[k]] = 0
        else:
            i, j, k_ = I[k], J[k], K[k]
            indOut[J[k] + 1, I[k], K[k]] = 0
        ai = 2 * phijik[j + 1, i, k_] * phijik[j - 1, i, k_]
        aim1 = -phijik[j, i, k_] * phijik[j + 1, i, k_]
        aip1 = -phijik[j, i, k_] * phijik[j - 1, i, k_]
        phiS = ai**2 + aim1**2 + aip1**2
        AddMat(rav(i, j - 1, k_), i, j - 1, k_, aim1 * aim1 / phiS)
        AddMat(rav(i, j - 1, k_), i, j, k_, aim1 * ai / phiS)
        AddMat(rav(i, j - 1, k_), i, j + 1, k_, aim1 * aip1 / phiS)
        AddMat(rav(i, j, k_), i, j - 1, k_, ai * aim1 / phiS)
        AddMat(rav(i, j, k_), i, j, k_, ai * ai / phiS)
        AddMat(rav(i, j, k_), i, j + 1, k_, ai * aip1 / phiS)
        AddMat(rav(i, j + 1, k_), i, j - 1, k_, aip1 * aim1 / phiS)
        AddMat(rav(i, j + 1, k_), i, j, k_, aip1 * ai / phiS)
        AddMat(rav(i, j + 1, k_), i, j + 1, k_, aip1 * aip1 / phiS)

    indz = ind[:, :, 1 : Nz + 1] - ind[:, :, 0:Nz]
    J, I, K = np.where((indz == 1) | (indz == -1))
    for k in range(np.shape(I)[0]):
        if indz[J[k], I[k], K[k]] == 1:
            i, j, k_ = I[k], J[k], K[k] + 1
            indOut[J[k], I[k], K[k]] = 0
        else:
            i, j, k_ = I[k], J[k], K[k]
            indOut[J[k], I[k], K[k] + 1] = 0
        ai = 2 * phijik[j, i, k_ + 1] * phijik[j, i, k_ - 1]
        aim1 = -phijik[j, i, k_] * phijik[j, i, k_ + 1]
        aip1 = -phijik[j, i, k_] * phijik[j, i, k_ - 1]
        phiS = ai**2 + aim1**2 + aip1**2
        AddMat(rav(i, j, k_ - 1), i, j, k_ - 1, aim1 * aim1 / phiS)
        AddMat(rav(i, j, k_ - 1), i, j, k_, aim1 * ai / phiS)
        AddMat(rav(i, j, k_ - 1), i, j, k_ + 1, aim1 * aip1 / phiS)
        AddMat(rav(i, j, k_), i, j, k_ - 1, ai * aim1 / phiS)
        AddMat(rav(i, j, k_), i, j, k_, ai * ai / phiS)
        AddMat(rav(i, j, k_), i, j, k_ + 1, ai * aip1 / phiS)
        AddMat(rav(i, j, k_ + 1), i, j, k_ - 1, aip1 * aim1 / phiS)
        AddMat(rav(i, j, k_ + 1), i, j, k_, aip1 * ai / phiS)
        AddMat(rav(i, j, k_ + 1), i, j, k_ + 1, aip1 * aip1 / phiS)

    npcoef = (gamma / hx / hy) * np.array(coef)
    B = sp.coo_array((npcoef, (row, col)), shape=(Ndof, Ndof))

    def rav(i, j, k):
        return np.ravel_multi_index([j, i, k], (Ny + 1, Nx + 1, Nz + 1))

    def AddMat(eq, i, j, k, a):
        row.append(eq)
        col.append(rav(i, j, k))
        coef.append(a)

    indx = ind[:, 1 : Nx + 1, :] - ind[:, 0:Nx, :]
    J, I, K = np.where((indx == 1) | (indx == -1))
    for k in range(np.shape(I)[0]):
        if indx[J[k], I[k], K[k]] == 1:
            ii, j, k_ = I[k] + 1, J[k], K[k]
        else:
            ii, j, k_ = I[k] - 1, J[k], K[k]
        if (
            ind[j, ii - 1, k_] == 0
            and ind[j, ii, k_] == 1
            and ind[j, ii + 1, k_] == 1
            and ind[j, ii + 2, k_] == 1
        ) or (
            ind[j, ii - 1, k_] == 1
            and ind[j, ii, k_] == 1
            and ind[j, ii + 1, k_] == 1
            and ind[j, ii + 2, k_] == 0
        ):
            AddMat(rav(ii - 1, j, k_), ii - 1, j, k_, 1.0)
            AddMat(rav(ii - 1, j, k_), ii, j, k_, -3.0)
            AddMat(rav(ii - 1, j, k_), ii + 1, j, k_, 3.0)
            AddMat(rav(ii - 1, j, k_), ii + 2, j, k_, -1.0)
            AddMat(rav(ii, j, k_), ii - 1, j, k_, -3.0)
            AddMat(rav(ii, j, k_), ii, j, k_, 9.0)
            AddMat(rav(ii, j, k_), ii + 1, j, k_, -9.0)
            AddMat(rav(ii, j, k_), ii + 2, j, k_, 3.0)
            AddMat(rav(ii + 1, j, k_), ii - 1, j, k_, 3.0)
            AddMat(rav(ii + 1, j, k_), ii, j, k_, -9.0)
            AddMat(rav(ii + 1, j, k_), ii + 1, j, k_, 9.0)
            AddMat(rav(ii + 1, j, k_), ii + 2, j, k_, -3.0)
            AddMat(rav(ii + 2, j, k_), ii - 1, j, k_, -1.0)
            AddMat(rav(ii + 2, j, k_), ii, j, k_, 3.0)
            AddMat(rav(ii + 2, j, k_), ii + 1, j, k_, -3.0)
            AddMat(rav(ii + 2, j, k_), ii + 2, j, k_, 1.0)

    indy = ind[1 : Ny + 1, :, :] - ind[0:Ny, :, :]
    J, I, K = np.where((indy == 1) | (indy == -1))
    for k in range(np.shape(I)[0]):
        if indy[J[k], I[k], K[k]] == 1:
            i, jj, k_ = I[k], J[k] + 1, K[k]
        else:
            i, jj, k_ = I[k], J[k] - 1, K[k]
        if (
            ind[jj - 1, i, k_] == 0
            and ind[jj, i, k_] == 1
            and ind[jj + 1, i, k_] == 1
            and ind[jj + 2, i, k_] == 1
        ) or (
            ind[jj - 1, i, k_] == 1
            and ind[jj, i, k_] == 1
            and ind[jj + 1, i, k_] == 1
            and ind[jj + 2, i, k_] == 0
        ):
            AddMat(rav(i, jj - 1, k_), i, jj - 1, k_, 1.0)
            AddMat(rav(i, jj - 1, k_), i, jj, k_, -3.0)
            AddMat(rav(i, jj - 1, k_), i, jj + 1, k_, 3.0)
            AddMat(rav(i, jj - 1, k_), i, jj + 2, k_, -1.0)
            AddMat(rav(i, jj, k_), i, jj - 1, k_, -3.0)
            AddMat(rav(i, jj, k_), i, jj, k_, 9.0)
            AddMat(rav(i, jj, k_), i, jj + 1, k_, -9.0)
            AddMat(rav(i, jj, k_), i, jj + 2, k_, 3.0)
            AddMat(rav(i, jj + 1, k_), i, jj - 1, k_, 3.0)
            AddMat(rav(i, jj + 1, k_), i, jj, k_, -9.0)
            AddMat(rav(i, jj + 1, k_), i, jj + 1, k_, 9.0)
            AddMat(rav(i, jj + 1, k_), i, jj + 2, k_, -3.0)
            AddMat(rav(i, jj + 2, k_), i, jj - 1, k_, -1.0)
            AddMat(rav(i, jj + 2, k_), i, jj, k_, 3.0)
            AddMat(rav(i, jj + 2, k_), i, jj + 1, k_, -3.0)
            AddMat(rav(i, jj + 2, k_), i, jj + 2, k_, 1.0)

    indz = ind[:, :, 1 : Nz + 1] - ind[:, :, 0:Nz]
    J, I, K = np.where((indz == 1) | (indz == -1))
    for k in range(np.shape(I)[0]):
        if indz[J[k], I[k], K[k]] == 1:
            i, j, k__ = I[k], J[k], K[k] + 1
        else:
            i, j, k__ = I[k], J[k], K[k] - 1
        if (
            ind[j, i, k__ - 1] == 0
            and ind[j, i, k__] == 1
            and ind[j, i, k__ + 1] == 1
            and ind[j, i, k__ + 2] == 1
        ) or (
            ind[j, i, k__ - 1] == 1
            and ind[j, i, k__] == 1
            and ind[j, i, k__ + 1] == 1
            and ind[j, i, k__ + 2] == 0
        ):
            AddMat(rav(i, j, k__ - 1), i, j, k__ - 1, 1.0)
            AddMat(rav(i, j, k__ - 1), i, j, k__, -3.0)
            AddMat(rav(i, j, k__ - 1), i, j, k__ + 1, 3.0)
            AddMat(rav(i, j, k__ - 1), i, j, k__ + 2, -1.0)
            AddMat(rav(i, j, k__), i, j, k__ - 1, -3.0)
            AddMat(rav(i, j, k__), i, j, k__, 9.0)
            AddMat(rav(i, j, k__), i, j, k__ + 1, -9.0)
            AddMat(rav(i, j, k__), i, j, k__ + 2, 3.0)
            AddMat(rav(i, j, k__ + 1), i, j, k__ - 1, 3.0)
            AddMat(rav(i, j, k__ + 1), i, j, k__, -9.0)
            AddMat(rav(i, j, k__ + 1), i, j, k__ + 1, 9.0)
            AddMat(rav(i, j, k__ + 1), i, j, k__ + 2, -3.0)
            AddMat(rav(i, j, k__ + 2), i, j, k__ - 1, -1.0)
            AddMat(rav(i, j, k__ + 2), i, j, k__, 3.0)
            AddMat(rav(i, j, k__ + 2), i, j, k__ + 1, -3.0)
            AddMat(rav(i, j, k__ + 2), i, j, k__ + 2, 1.0)

    npcoef = sigma / (hx * hy) * np.array(coef)
    C = sp.coo_array((npcoef, (row, col)), shape=(Ndof, Ndof))

    # penalization outside
    D = sp.diags(diagonals=indOut.ravel())

    # linear system
    A = (A + B + C + D).tocsr()
    b = f(X, Y, Z)
    b = (ind * b).ravel()
    u = spsolve(A, b).reshape(Ny + 1, Nx + 1, Nz + 1)
    # final time
    t_final = time()
    uref = ue(X, Y, Z)
    e = u - uref

    eL2 = np.sqrt(np.sum(e * e * ind)) / np.sqrt(np.sum(uref * uref * ind))
    emax = np.max(np.abs(e * ind)) / np.max(np.abs(uref * ind))
    ex = (e[:, 1 : Nx + 1, :] - e[:, 0:Nx, :]) / hx
    urefx = (uref[:, 1 : Nx + 1, :] - uref[:, 0:Nx, :]) / hx
    intdx = (ind[:, 1 : Nx + 1, :] + ind[:, 0:Nx, :] == 2) + 0
    fulldx = (ind[:, 1 : Nx + 1, :] + ind[:, 0:Nx, :] > 0) + 0

    ey = (e[1 : Ny + 1, :, :] - e[0:Ny, :, :]) / hy
    urefy = (uref[1 : Ny + 1, :, :] - uref[0:Ny, :, :]) / hy
    intdy = (ind[1 : Ny + 1, :, :] + ind[0:Ny, :, :] == 2) + 0
    fulldy = (ind[1 : Ny + 1, :, :] + ind[0:Ny, :, :] > 0) + 0

    ez = (e[:, :, 1 : Nz + 1] - e[:, :, 0:Nz]) / hz
    urefz = (uref[:, :, 1 : Nz + 1] - uref[:, :, 0:Nz]) / hz
    intdz = (ind[:, :, 1 : Nz + 1] + ind[:, :, 0:Nz] == 2) + 0
    fulldz = (ind[:, :, 1 : Nz + 1] + ind[:, :, 0:Nz] > 0) + 0

    eH1 = np.sqrt(
        (np.sum(ex * ex * fulldx) + np.sum(ey * ey * fulldy) + np.sum(ez * ez * fulldz))
    ) / np.sqrt(
        (
            np.sum(urefx * urefx * fulldx)
            + np.sum(urefy * urefy * fulldy)
            + np.sum(urefz * urefz * fulldz)
        )
    )
    eH1int = np.sqrt(
        (np.sum(ex * ex * intdx) + np.sum(ey * ey * intdy) + np.sum(ez * ez * intdz))
    ) / np.sqrt(
        (
            np.sum(urefx * urefx * intdx)
            + np.sum(urefy * urefy * intdy)
            + np.sum(urefz * urefz * intdz)
        )
    )

    print("h :", np.sqrt(2) / N)
    print("relative L2 error diff : ", eL2)
    print("relative L inf error diff : ", emax)
    print("relative H1 error diff : ", eH1)
    print("relative H1 error diff int : ", eH1int)
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


plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.loglog(size_mesh_standard_vec, error_L2_standard_vec, "-+", label="L2 std")
plt.loglog(size_mesh_phi_vec, error_L2_phifem_vec, "-+", label="L2 phiFEM")
plt.loglog(size_mesh_phi_vec, error_L2_diff_vec, "-+", label="L2 phiFD")
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

plt.savefig("errors.pdf")
plt.show()


def order(a, b):
    return -np.polyfit(np.log(a), -np.log(b), 1)[0]


order_L2_phifem = order(size_mesh_phi_vec, error_L2_phifem_vec)
order_Linf_phifem = order(size_mesh_phi_vec, error_Linf_phifem_vec)
order_H1int_phifem = order(size_mesh_phi_vec, error_H1int_phifem_vec)
order_L2_standard = order(size_mesh_standard_vec, error_L2_standard_vec)
order_Linf_standard = order(size_mesh_standard_vec, error_Linf_standard_vec)
order_H1int_standard = order(size_mesh_standard_vec, error_H1int_standard_vec)
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

f.write("relative L2 norm for phifem, FEM standard, phi-FD, phi-FD2: \n")
f.write("\\addplot[mark=*, blue] coordinates {\n")
output_latex(f, size_mesh_phi_vec, error_L2_phifem_vec)
f.write("};\n\\addplot[mark=*] coordinates {\n")
output_latex(f, size_mesh_standard_vec, error_L2_standard_vec)
f.write("};\n\\addplot[mark=*,red] coordinates {\n")
output_latex(f, size_mesh_phi_vec, error_L2_diff_vec)
f.write("};\n\\addplot[mark=*,orange] coordinates {\n")
output_latex(f, size_mesh_phi_vec, error_L2_diff2_vec)
f.write("};\n\n")


f.write("relative L infty norm for phifem, FEM standard, phi-FD, phi-FD2: \n")
f.write("\\addplot[mark=*, blue] coordinates {\n")
output_latex(f, size_mesh_phi_vec, error_Linf_phifem_vec)
f.write("};\n\\addplot[mark=*] coordinates {\n")
output_latex(f, size_mesh_standard_vec, error_Linf_standard_vec)
f.write("};\n\\addplot[mark=*,red] coordinates {\n")
output_latex(f, size_mesh_phi_vec, error_Linf_diff_vec)
f.write("};\n\\addplot[mark=*,orange] coordinates {\n")
output_latex(f, size_mesh_phi_vec, error_Linf_diff2_vec)
f.write("};\n\n")


f.write(
    "relative H1 norm in Omega_h^Gamma for phifem, FEM standard, phi-FD, phi-FD2: \n"
)
f.write("\\addplot[mark=*, blue] coordinates {\n")
output_latex(f, size_mesh_phi_vec, error_H1int_phifem_vec)
f.write("};\n\\addplot[mark=*] coordinates {\n")
output_latex(f, size_mesh_standard_vec, error_H1int_standard_vec)
f.write("};\n\\addplot[mark=*,red] coordinates {\n")
output_latex(f, size_mesh_phi_vec, error_H1int_diff_vec)
f.write("};\n\\addplot[mark=*,orange] coordinates {\n")
output_latex(f, size_mesh_phi_vec, error_H1int_diff2_vec)
f.write("};\n\n")


f.write(
    "relative L2 norm and time standard for phifem, FEM standard, phi-FD, phi-FD2: \n"
)
f.write("\\addplot[mark=*, blue] coordinates {\n")
output_latex(f, error_L2_phifem_vec, time_phifem_vec)
f.write("};\n\\addplot[mark=*] coordinates {\n")
output_latex(f, error_L2_standard_vec, time_standard_vec)
f.write("};\n\\addplot[mark=*,red] coordinates {\n")
output_latex(f, error_L2_diff_vec, time_diff_vec)
f.write("};\n\\addplot[mark=*,orange] coordinates {\n")
output_latex(f, error_L2_diff2_vec, time_diff2_vec)
f.write("};\n\n")

f.write("relative Linf norm and time for phifem, FEM standard, phi-FD, phi-FD2: \n")
f.write("\\addplot[mark=*, blue] coordinates {\n")
output_latex(f, error_Linf_phifem_vec, time_phifem_vec)
f.write("};\n\\addplot[mark=*] coordinates {\n")
output_latex(f, error_Linf_standard_vec, time_standard_vec)
f.write("};\n\\addplot[mark=*,red] coordinates {\n")
output_latex(f, error_Linf_diff_vec, time_diff_vec)
f.write("};\n\\addplot[mark=*,orange] coordinates {\n")
output_latex(f, error_Linf_diff2_vec, time_diff2_vec)
f.write("};\n\n")

f.write(
    "relative H1 norm in Omega_h^Gamma and time for phifem, FEM standard, phi-FD, phi-FD2: \n"
)
f.write("\\addplot[mark=*, blue] coordinates {\n")
output_latex(f, error_H1int_phifem_vec, time_phifem_vec)
f.write("};\n\\addplot[mark=*] coordinates {\n")
output_latex(f, error_H1int_standard_vec, time_standard_vec)
f.write("};\n\\addplot[mark=*,red] coordinates {\n")
output_latex(f, error_H1int_diff_vec, time_diff_vec)
f.write("};\n\\addplot[mark=*,orange] coordinates {\n")
output_latex(f, error_H1int_diff2_vec, time_diff2_vec)
f.write("};\n\n")

if conditioning == True:
    f.write("conditioning number for phifem, FEM standard, phi-FD, phi-FD2: \n")
    f.write("\\addplot[mark=*, blue] coordinates {\n")
    output_latex(f, size_mesh_phi_vec, cond_phifem_vec)
    f.write("};\n\\addplot[mark=*] coordinates {\n")
    output_latex(f, size_mesh_standard_vec, cond_standard_vec)
    f.write("};\n\\addplot[mark=*,red] coordinates {\n")
    output_latex(f, size_mesh_phi_vec, cond_diff_vec)
    f.write("};\n\\addplot[mark=*,orange] coordinates {\n")
    output_latex(f, size_mesh_phi_vec, cond_diff2_vec)
    f.write("};\n\n")

f.write("Order of convergence in L2\n")
f.write(str(round(order_L2_phifem, 2)))
f.write("&")
f.write(str(round(order_L2_standard, 2)))
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
f.write(str(round(order_Linf_diff, 2)))
f.write("&")
f.write(str(round(order_Linf_diff2, 2)))
f.write("\\\\\n\n")

f.write("Order of convergence in H1\n")
f.write(str(round(order_H1int_phifem, 2)))
f.write("&")
f.write(str(round(order_H1int_standard, 2)))
f.write("&")
f.write(str(round(order_H1int_diff, 2)))
f.write("&")
f.write(str(round(order_H1int_diff2, 2)))
f.write("\\\\\n\n")

f.close()
