import numpy as np
from scipy.optimize import curve_fit


def catenary_curve(u: np.ndarray, x0: float, y0: float, c: float) -> np.ndarray:
    """Evaluate the 2D catenary equation."""
    return y0 + c * np.cosh((u - x0) / c) - c


def fit_catenary(local_points: np.ndarray) -> tuple[float, float, float]:
    """Fit a catenary curve to local 2D wire points."""
    u = local_points[:, 0]
    v = local_points[:, 1]

    x0_guess = u[np.argmin(v)]
    y0_guess = np.min(v)
    c_guess = max((np.max(u) - np.min(u)) / 4, 1.0)

    params, _ = curve_fit(
        catenary_curve,
        u,
        v,
        p0=[x0_guess, y0_guess, c_guess],
        maxfev=10000,
    )

    x0, y0, c = params
    return x0, y0, c
