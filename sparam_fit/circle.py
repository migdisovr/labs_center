"""Algebraic circle fit of Chernov & Lesort (JMIV 2005), as used by Probst et al."""

from __future__ import annotations

import numpy as np
from scipy.optimize import newton


def fit_circle_algebraic(z, refine: bool = False):
    """Fit A(x^2+y^2)+Bx+Cy+D=0 with B^2+C^2-4AD=1.

    Returns ``(xc, yc, r0)``.  Data may be translated internally for stability;
    the returned centre is in the original coordinates.
    """
    z = np.asarray(z, dtype=np.complex128)
    mean = np.mean(z)
    zc = z - mean
    x, y = zc.real, zc.imag
    zz = x * x + y * y
    n = float(len(z))

    Mxx = np.sum(x * x)
    Myy = np.sum(y * y)
    Mxy = np.sum(x * y)
    Mxz = np.sum(x * zz)
    Myz = np.sum(y * zz)
    Mzz = np.sum(zz * zz)
    Mx = np.sum(x)
    My = np.sum(y)
    Mz = np.sum(zz)

    M = np.array(
        [
            [Mzz, Mxz, Myz, Mz],
            [Mxz, Mxx, Mxy, Mx],
            [Myz, Mxy, Myy, My],
            [Mz, Mx, My, n],
        ],
        dtype=float,
    )
    Bmat = np.array(
        [
            [0.0, 0.0, 0.0, -2.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-2.0, 0.0, 0.0, 0.0],
        ]
    )

    def xi(eta):
        return np.linalg.det(M - eta * Bmat)

    def dxi(eta):
        A = M - eta * Bmat
        # d/deta det(A) = det(A) Tr(A^{-1} dA/deta)
        try:
            return float(np.linalg.det(A) * np.trace(-np.linalg.solve(A, Bmat)))
        except np.linalg.LinAlgError:
            return np.nan

    try:
        eta = float(newton(xi, 0.0, fprime=dxi, maxiter=50, tol=1e-14))
    except (RuntimeError, OverflowError):
        eta = 0.0
        best = abs(xi(0.0))
        for guess in np.linspace(0.0, 1.0, 40):
            val = abs(xi(guess))
            if val < best:
                best, eta = val, float(guess)

    Aeta = M - eta * Bmat
    _, _, Vt = np.linalg.svd(Aeta)
    avec = Vt[-1]
    # Enforce Chernov constraint B^2 + C^2 - 4AD = 1
    A, B, C, D = avec
    cons = B * B + C * C - 4.0 * A * D
    if cons <= 0:
        # Fall back to Kasa-like geometric estimate
        xc = float(np.mean(x))
        yc = float(np.mean(y))
        r0 = float(np.mean(np.sqrt((x - xc) ** 2 + (y - yc) ** 2)))
    else:
        avec = avec / np.sqrt(cons)
        A, B, C, D = avec
        xc = -B / (2.0 * A)
        yc = -C / (2.0 * A)
        r0 = 1.0 / (2.0 * abs(A))

    if refine:
        xc, yc, r0 = _refine_circle(x, y, xc, yc, r0)

    return float(xc + mean.real), float(yc + mean.imag), float(r0)


def _refine_circle(x, y, xc, yc, r0):
    from scipy.optimize import least_squares

    def resid(p):
        return np.sqrt((x - p[0]) ** 2 + (y - p[1]) ** 2) - p[2]

    res = least_squares(resid, [xc, yc, r0], xtol=1e-12, ftol=1e-12)
    return float(res.x[0]), float(res.x[1]), float(abs(res.x[2]))


def circle_chi2(z, xc, yc, r0):
    z = np.asarray(z, dtype=np.complex128)
    r = np.sqrt((z.real - xc) ** 2 + (z.imag - yc) ** 2)
    return float(np.mean((r - r0) ** 2))
