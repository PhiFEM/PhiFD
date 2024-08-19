from typing import Any
import numpy as np
import pandas as pd
import scipy.sparse.linalg as sla
import time
from matplotlib import pyplot as plt
import scipy.sparse as sps

pp = print


def linear_direct(A, b):
    u = sla.spsolve(A, b)
    return u


class Iterative_info:
    def __init__(self, r_tol, error_fn, maxiter):
        self.status = "scipy_stop"
        self.u = None
        self.end = time.time()
        self.iter = 0
        self.error_fn = error_fn
        self.r_tol = r_tol
        self.maxiter = maxiter
        self.residues = []
        self.errors = []
        self.best_u = None
        self.final_iter = 0

    def __call__(self, u):
        self.u = u
        end = time.time()
        residuals = self.error_fn(u)
        self.residues.append(residuals)
        self.iter += 1
        if residuals == min(self.residues):
            self.best_u = self.u
            self.end = end
            self.final_iter = self.iter

        if residuals < self.r_tol:
            self.status = "error_max_stop"
            raise StopIteration
        if self.maxiter != None:
            if self.iter >= self.maxiter:
                self.status = "max iter"
                raise StopIteration


def iterative_solver_controled(
    A,
    b,
    u0,
    error_fn,
    r_tol=1e-14,
    maxiter=None,
):
    ite_info = Iterative_info(r_tol, error_fn, maxiter)
    try:
        sla.bicgstab(A, b, x0=u0, rtol=1e-10, callback=ite_info, maxiter=maxiter)
    except StopIteration:
        pass
    return ite_info.best_u, ite_info


class ErrorFnAb_relative:
    def __init__(self, A, b, ind, mesh):
        self.A = A
        self.b = b
        self.ind = ind
        self.mesh = mesh
        self.rhs = self.ind * self.b
        self.magn = np.sqrt(np.sum(self.ind * self.b))

    def __call__(self, u):
        e = self.ind * (self.A @ u) - self.rhs
        return np.sqrt(np.sum(e**2)) / self.magn


class ErrorL2:
    def __init__(self, A, b, u_, ind, mesh, uref):
        self.A = A
        self.b = b
        self.ind = ind
        self.mesh = mesh
        self.u_ = u_
        self.rhs = self.ind * self.b
        self.magn = np.sqrt(np.sum(self.ind * self.b))
        self.uref = uref(self.mesh.X, self.mesh.Y)

    def __call__(self, u):
        e = self.ind * (u + self.u_ - self.uref)
        eL2 = np.sqrt(np.sum(e * e)) / np.sqrt(np.sum(self.uref * self.uref * self.ind))
        return eL2
