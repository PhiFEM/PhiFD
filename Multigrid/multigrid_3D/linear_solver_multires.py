from typing import Any
import scipy
import scipy.sparse as sps
import scipy.sparse.linalg as sla
import time
from linear_syst import (
    linear_direct,
    iterative_solver_controled,
    ErrorFnAb_relative2,
    ErrorFnAb_relative,
    ErrorL2,
)
from phiFD import Mesh, build_matrices, force
import scipy.ndimage as sndimage
import numpy as np
import matplotlib.pyplot as plt


pp = print


def interpolation_via_reshape(img_in: Any, input_size, output_size, order, prefilter):
    img_in = np.reshape(img_in, [input_size, input_size, input_size])

    zoom = output_size / input_size
    img_out = sndimage.zoom(
        img_in,
        zoom=zoom,
        order=order,
        prefilter=prefilter,
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
    r_tol=1e-6,
    order=2,
):

    N_coarse = min(100, N)
    if maxiter == None:
        maxiter = max(20, (N - N_coarse) // 5)
    print(f"maxiter = {maxiter}")
    N_iterative = N_final

    mesh_fine = Mesh(N_iterative)
    ind_fine, A_fine = build_matrices(mesh_fine, phi)
    b_fine = force(mesh_fine, ind_fine, f)

    if explicit_expression:
        assert callable(phi)
    else:
        if callable(phi):
            phi_fine = phi(mesh_fine.X, mesh_fine.Y, mesh_fine.Z)
            phi = interpolation_via_reshape(
                phi_fine, N_iterative + 1, N_coarse + 1, order=order, prefilter=True
            ).reshape((N_coarse + 1, N_coarse + 1, N_coarse + 1))
            b_coarse = interpolation_via_reshape(
                b_fine, N_iterative + 1, N_coarse + 1, order=order, prefilter=True
            )

    mesh_coarse = Mesh(N_coarse)
    ind_coarse, A_coarse = build_matrices(mesh_coarse, phi)
    if explicit_expression:
        b_coarse = force(mesh_coarse, ind_coarse, f)

    error_fn_coarse = ErrorFnAb_relative(A_coarse, b_coarse, ind_coarse, mesh_coarse)
    start = time.time()
    u_coarse, info = iterative_solver_controled(
        A_coarse,
        b_coarse,
        None,
        error_fn_coarse,
        maxiter=maxiter * 4,
        r_tol=1e-6,
    )

    u_0 = interpolation_via_reshape(
        u_coarse, N_coarse + 1, N_iterative + 1, order=order, prefilter=True
    )
    end = time.time()
    time_coarse = end - start
    print(f"First solver: Stop after {info.iter} iterations")
    print(f"Best u after {info.final_iter} iterations")

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
    time_fine = end - start
    print(f"Final solver: Stop after {info.iter} iterations")
    print(f"Best u after {info.final_iter} iterations")

    return u, ind_fine, mesh_fine, time_coarse + time_fine
