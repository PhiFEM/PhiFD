from typing import Any
import time
from linear_syst import (
    linear_direct,
    iterative_solver_controled,
    ErrorFnAb_relative,
)
from phiFD import Mesh, build_matrices, force
import scipy.ndimage as sndimage
import numpy as np


pp = print


def interpolation_via_reshape(img_in: Any, input_size, output_size, order, prefilter):
    img_in = np.reshape(img_in, [input_size, input_size])

    zoom = output_size / input_size
    img_out = sndimage.zoom(
        img_in,
        zoom=zoom,
        order=order,
        prefilter=prefilter,
        grid_mode=False,
        mode="constant",
    )

    return img_out.flatten()


def solve_multiRes(
    N: int,
    N_final: int,
    phi,
    f,
    explicit_expression=False,
    maxiter=None,
    r_tol=1e-4,
    order=2,
):
    N_direct = min(1000, N)
    if maxiter == None:
        maxiter = max(20, (N - N_direct) // 5)
    print(f"maxiter = {maxiter}")
    N_iterative = N_final

    mesh_fine = Mesh(N_iterative)
    ind_fine, A_fine = build_matrices(mesh_fine, phi)
    b_fine = force(mesh_fine, ind_fine, f)

    if explicit_expression:
        assert callable(phi)
    else:
        if callable(phi):
            phi_fine = phi(mesh_fine.X, mesh_fine.Y)
            phi = interpolation_via_reshape(
                phi_fine, N_iterative + 1, N_direct + 1, order=order, prefilter=True
            ).reshape((N_direct + 1, N_direct + 1))
            b_coarse = interpolation_via_reshape(
                b_fine, N_iterative + 1, N_direct + 1, order=order, prefilter=True
            )

    mesh_direct = Mesh(N_direct)
    ind_direct, A_direct = build_matrices(mesh_direct, phi)
    if explicit_expression:
        b_coarse = force(mesh_direct, ind_direct, f)
    start = time.time()
    u_coarse = linear_direct(A_direct, b_coarse)

    u_0 = interpolation_via_reshape(
        u_coarse, N_direct + 1, N_iterative + 1, order=order, prefilter=True
    )
    end = time.time()
    time_direct = end - start

    error_fn_fine = ErrorFnAb_relative(A_fine, b_fine, ind_fine, mesh_fine)
    start = time.time()
    u, info = iterative_solver_controled(
        A_fine,
        b_fine,
        u_0,
        error_fn_fine,
        maxiter=maxiter,
        r_tol=r_tol,
    )
    end = info.end  # time.time()
    time_iterative = end - start
    print(f"Stop after {info.iter} iterations with initial guess u0")
    print(f"Best u after {info.final_iter} iterations with initial guess u0")

    return u, ind_fine, mesh_fine, time_direct + time_iterative
