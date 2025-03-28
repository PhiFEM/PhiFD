import os
import pickle
import time
from linear_solver_multires import interpolation_via_reshape
from linear_syst import linear_direct, iterative_solver_controled, ErrorFnAb_relative
from phiFD import (
    problem_with_solution,
    Mesh,
    build_matrices,
    force,
    errors_L2_Loo_fn,
)

pp = print


def compute_ref_quantity(N, test_case=None, iterative=False):
    if test_case == None:
        test_case = "circle_cos"
    phi, f, uref_fn = problem_with_solution(test_case)

    if iterative:
        path = f"./ref_quantities_iterative/ref_quant_{test_case}{N}"
    else:
        path = f"./ref_quantities_direct/ref_quant_{test_case}{N}"
    if os.path.exists(path):
        res = pickle.load(open(path, "rb"))
    else:
        if iterative:
            print(f"compute iterative {test_case} ref quantities for N={N} ")
            if not os.path.exists("./ref_quantities_iterative"):
                os.makedirs("./ref_quantities_iterative")
        else:
            print(f"compute direct {test_case} ref quantities for N={N} ")
            if not os.path.exists("./ref_quantities_direct"):
                os.makedirs("./ref_quantities_direct")

        mesh = Mesh(N)
        ind, A = build_matrices(mesh, phi)
        b = force(mesh, ind, f)
        if iterative:
            error_fn = ErrorFnAb_relative(A, b, ind, mesh)
            ti0 = time.time()
            u, info = iterative_solver_controled(
                A,
                b,
                None,
                error_fn,
                r_tol=1e-4,
                maxiter=10000,
            )
            end_solve = time.time()

            print(f"Iterative solver: {info.status} after {info.iter} iterations")
        else:
            ti0 = time.time()
            u = linear_direct(A, b)
            end_solve = time.time()
        uref = uref_fn(mesh.X, mesh.Y)
        errors = errors_L2_Loo_fn(ind, u, uref, mesh)
        res = {
            "duration": end_solve - ti0,
            "L2": errors["L2"],
            "Loo": errors["Loo"],
            "uref": uref,
            "u": u,
            "ind": ind,
        }
        pickle.dump(res, open(path, "wb"))
    return res


class RefAgent:
    def __init__(self, N, N_final=2200, test_case=None, iterative=False):
        quantity_N = compute_ref_quantity(N, test_case, iterative=iterative)
        quantity_N_obs = compute_ref_quantity(N_final, test_case)
        u_N = quantity_N["u"]
        ind_N_obs = quantity_N_obs["ind"]
        uref_N_obs = quantity_N_obs["uref"]
        mesh_N_obs = Mesh(N_final)
        u_N = quantity_N["u"]
        if N_final != N:
            ti0 = time.time()
            u_N_to_Nlast = interpolation_via_reshape(u_N, N + 1, N_final + 1, 1, True)
            dur = time.time() - ti0
        else:
            u_N_to_Nlast = u_N
            dur = 0
        errors_Loo_L2 = errors_L2_Loo_fn(
            ind_N_obs, u_N_to_Nlast, uref_N_obs, mesh_N_obs
        )
        dur += quantity_N["duration"]
        self.result = {
            "duration": dur,
            "Loo": errors_Loo_L2["Loo"],
            "L2": errors_Loo_L2["L2"],
        }
