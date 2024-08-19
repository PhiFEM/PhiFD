from linear_solver_multires import solve_multiRes
from phiFD import problem_with_solution, errors_L2_Loo_fn


class Solver_multigrid:
    def __init__(
        self,
        N: int,
        N_final: int,
        test_case: str,
        explicit_expression: bool,
        maxiter=None,
        r_tol=1e-4,
        order=2,
    ):

        self.test_case = test_case
        self.phi, self.f, self.uref_fn = problem_with_solution(self.test_case)
        self.results = {}
        self.N = N
        self.N_final = N_final
        self.explicit_expression = explicit_expression
        self.maxiter = maxiter
        self.r_tol = r_tol
        self.order = order

    def solve(self) -> dict:

        u, ind, mesh, time_multires = solve_multiRes(
            self.N,
            self.N_final,
            self.phi,
            self.f,
            explicit_expression=self.explicit_expression,
            maxiter=self.maxiter,
            r_tol=self.r_tol,
            order=self.order,
        )
        duration = time_multires
        uref = self.uref_fn(mesh.X, mesh.Y)
        errors = errors_L2_Loo_fn(ind, u, uref, mesh)
        self.results = {
            "L2": errors["L2"],
            "duration": duration,
            "Loo": errors["Loo"],
        }
        return self.results
