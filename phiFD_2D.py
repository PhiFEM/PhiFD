import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

# Radius of the domain
R = 0.3 + 1e-10

# Parameter of penalisation and stabilisation
sigma, gamma = 0.01, 1.0

# Construction of the grid
Nx, Ny = 100, 100
x, y = np.linspace(0, 1, Nx + 1), np.linspace(0, 1, Ny + 1)
hx, hy = x[1] - x[0], y[1] - y[0]
X, Y = np.meshgrid(x, y)

# Computation of the exact solution, exact source term and the levelset
r = lambda x, y: np.sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) + 1e-12)
K = np.pi / 2 / R
ue = lambda x, y: np.cos(K * r(x, y))
f = lambda x, y: K * K * np.cos(K * r(x, y)) + K * np.sin(K * r(x, y)) / r(x, y)
phi = lambda x, y: (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) - R * R
phiij = phi(X, Y)
ind = (phiij < 0) + 0
mask = sp.diags(diagonals=ind.ravel())
indOut = 1 - ind

# Laplacian matrix
D2x = (1 / hx / hx) * sp.diags(
    diagonals=[-1, 2, -1], offsets=[-1, 0, 1], shape=(Nx + 1, Nx + 1)
)
D2y = (1 / hy / hy) * sp.diags(
    diagonals=[-1, 2, -1], offsets=[-1, 0, 1], shape=(Ny + 1, Ny + 1)
)
D2x_2d = sp.kron(sp.eye(Ny + 1), D2x)
D2y_2d = sp.kron(D2y, sp.eye(Nx + 1))
A = mask @ (D2x_2d + D2y_2d)

# Boundary conditions
diag = np.zeros((Nx + 1) * (Ny + 1))
diagxp = np.zeros((Nx + 1) * (Ny + 1) - 1)
diagxm = np.zeros((Nx + 1) * (Ny + 1) - 1)
diagyp = np.zeros((Nx + 1) * Ny)
diagym = np.zeros((Nx + 1) * Ny)
actGx = np.zeros((Ny + 1, Nx + 1))
actGy = np.zeros((Ny + 1, Nx + 1))

indx = ind[:, 1 : Nx + 1] - ind[:, 0:Nx]
J, I = np.where((indx == 1) | (indx == -1))
for k in range(np.shape(I)[0]):
    if indx[J[k], I[k]] == 1:
        indOut[J[k], I[k]], actGx[J[k], I[k] + 1] = 0, 1
    else:
        indOut[J[k], I[k] + 1], actGx[J[k], I[k]] = 0, 1
phiS = np.square(phiij[J, I]) + np.square(phiij[J, I + 1])
diag[I + (Nx + 1) * J] = phiij[J, I + 1] * phiij[J, I + 1] / phiS
diagxp[I + (Nx + 1) * J] = -phiij[J, I] * phiij[J, I + 1] / phiS
diag[I + 1 + (Nx + 1) * J] = phiij[J, I] * phiij[J, I] / phiS
diagxm[I + (Nx + 1) * J] = -phiij[J, I] * phiij[J, I + 1] / phiS

indy = ind[1 : Ny + 1, :] - ind[0:Ny, :]
J, I = np.where((indy == 1) | (indy == -1))
for k in range(np.shape(I)[0]):
    if indy[J[k], I[k]] == 1:
        indOut[J[k], I[k]], actGy[J[k] + 1, I[k]] = 0, 1
    else:
        indOut[J[k] + 1, I[k]], actGy[J[k], I[k]] = 0, 1
phiS = np.square(phiij[J, I]) + np.square(phiij[J + 1, I])
diag[I + (Nx + 1) * J] += phiij[J + 1, I] * phiij[J + 1, I] / phiS
diagyp[I + (Nx + 1) * J] = -phiij[J, I] * phiij[J + 1, I] / phiS
diag[I + (Nx + 1) * (J + 1)] += phiij[J, I] * phiij[J, I] / phiS
diagym[I + (Nx + 1) * J] = -phiij[J, I] * phiij[J + 1, I] / phiS

B = (gamma / hx / hy) * sp.diags(
    diagonals=(diagym, diagxm, diag, diagxp, diagyp),
    offsets=(-Nx - 1, -1, 0, 1, Nx + 1),
)

# Stabilisation
maskGx = sp.diags(diagonals=actGx.ravel())
maskGy = sp.diags(diagonals=actGy.ravel())
C = sigma * hx * hy * (D2x_2d.T @ maskGx @ D2x_2d + D2y_2d.T @ maskGy @ D2y_2d)

# Penalization outside
D = sp.diags(diagonals=indOut.ravel())

# linear system
A, b = (A + B + C + D).tocsr(), (ind * f(X, Y)).ravel()
u = spsolve(A, b).reshape(Ny + 1, Nx + 1)

# Computation of the errors
uref = ue(X, Y)
e = ind * (u - uref)
eL2 = np.linalg.norm(e) * np.sqrt(hx * hy)
emax = np.linalg.norm(e, np.inf)
print(eL2, emax)
