"""Fit a bent cylinder to 3D points: polynomial centreline, constant radius; pure NumPy.

Centreline  c(t) = origin + t * u + A(t) * v + B(t) * w
where u is the main (PCA) direction of the points, v and w are perpendicular to
it, and A, B are polynomials of the chosen degree. The centreline is therefore a
graph over u: it can bend, but not turn back by 90 degrees or more.

Minimises  sum_i (|p_i - c(t_i*)| - r)^2  over the polynomial coefficients and r,
where c(t_i*) is the closest centreline point to p_i (true orthogonal distance).
Levenberg-Marquardt; the closest points are found by Gauss-Newton steps along
the curve, so by the envelope theorem the partial derivatives give the Jacobian.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class BentCylinder:
    origin: np.ndarray   # (3,) mm
    axes: np.ndarray     # (3, 3) rows u, v, w
    scale: float         # polynomials are evaluated in t / scale
    coef_a: np.ndarray   # A(t) coefficients, increasing degree
    coef_b: np.ndarray   # B(t) coefficients, increasing degree
    radius: float        # mm
    t_range: tuple       # centreline parameter range covered by the points, mm along u
    rms: float           # RMS distance of the points to the cylinder surface, mm

    def centreline(self, n=100):
        """(n, 3) points along the centreline over t_range."""
        return self._world(np.linspace(*self.t_range, n))

    def surface_distances(self, points):
        """Signed distance of each point to the cylinder surface (positive = outside), mm."""
        q = (np.asarray(points, dtype=float).reshape(-1, 3) - self.origin) @ self.axes.T
        t = _project(q, self.coef_a, self.coef_b, self.scale, q[:, 0].copy())
        return np.linalg.norm(q - _local(t, self.coef_a, self.coef_b, self.scale), axis=1) - self.radius

    def _world(self, t):
        return _local(t, self.coef_a, self.coef_b, self.scale) @ self.axes + self.origin


def fit_bent_cylinder(points, degree=3, *, max_points=5000, min_elongation=1.5, iterations=100):
    """Fit a BentCylinder to (N, 3) points in mm; None if the points are too few or not elongated.

    min_elongation: required std along the main axis relative to the largest
    perpendicular std; a ring from a single frame does not define an axis.
    max_points: larger clouds are subsampled evenly for speed.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    n_params = 2 * (degree + 1) + 1
    if len(points) < 3 * n_params:
        return None
    if len(points) > max_points:
        points = points[::int(np.ceil(len(points) / max_points))]
    origin = points.mean(axis=0)
    axes = np.linalg.svd(points - origin, full_matrices=False)[2]
    q = (points - origin) @ axes.T
    spread = q.std(axis=0)
    if spread[0] < min_elongation * max(spread[1], spread[2]):
        return None
    scale = float(np.abs(q[:, 0]).max())

    # Start: centreline through the middle of the points, radius = mean distance.
    t = q[:, 0].copy()
    basis = np.vander(t / scale, degree + 1, increasing=True)
    coef_a = np.linalg.lstsq(basis, q[:, 1], rcond=None)[0]
    coef_b = np.linalg.lstsq(basis, q[:, 2], rcond=None)[0]
    radius = np.linalg.norm(q[:, 1:] - np.column_stack([basis @ coef_a, basis @ coef_b]), axis=1).mean()
    theta = np.concatenate([coef_a, coef_b, [radius]])

    def evaluate(theta, t):
        a, b, r = theta[:degree + 1], theta[degree + 1:-1], theta[-1]
        t = _project(q, a, b, scale, t)
        basis = np.vander(t / scale, degree + 1, increasing=True)
        d = q - np.column_stack([t, basis @ a, basis @ b])
        rho = np.linalg.norm(d, axis=1)
        normal = d / np.maximum(rho, 1e-12)[:, None]
        jacobian = np.hstack([-normal[:, 1:2] * basis, -normal[:, 2:3] * basis, -np.ones((len(q), 1))])
        return rho - r, jacobian, t

    residuals, jacobian, t = evaluate(theta, t)
    cost, damping = residuals @ residuals, 1e-3
    for _ in range(iterations):
        hessian, gradient = jacobian.T @ jacobian, jacobian.T @ residuals
        step = np.linalg.solve(hessian + damping * np.diag(np.diag(hessian) + 1e-12), -gradient)
        trial = evaluate(theta + step, t)
        trial_cost = trial[0] @ trial[0]
        if trial_cost < cost:
            converged = cost - trial_cost < 1e-12 * max(cost, 1e-12)
            theta, (residuals, jacobian, t), cost = theta + step, trial, trial_cost
            damping = max(damping / 3, 1e-9)
            if converged:
                break
        else:
            damping *= 4
            if damping > 1e9:
                break

    return BentCylinder(origin=origin, axes=axes, scale=scale, coef_a=theta[:degree + 1],
                        coef_b=theta[degree + 1:-1], radius=float(abs(theta[-1])),
                        t_range=(float(t.min()), float(t.max())), rms=float(np.sqrt(cost / len(q))))


def _local(t, a, b, scale):
    basis = np.vander(t / scale, len(a), increasing=True)
    return np.column_stack([t, basis @ a, basis @ b])


def _project(q, a, b, scale, t, steps=10):
    """Parameter t of the closest centreline point to each q (local frame), Gauss-Newton."""
    powers = np.arange(len(a))
    for _ in range(steps):
        tau = t / scale
        basis = np.vander(tau, len(a), increasing=True)
        # d/dt of tau^k is k tau^(k-1) / scale
        d_basis = np.zeros_like(basis)
        d_basis[:, 1:] = basis[:, :-1] * powers[1:] / scale
        offset = q - np.column_stack([t, basis @ a, basis @ b])
        tangent = np.column_stack([np.ones_like(t), d_basis @ a, d_basis @ b])
        t = t + (offset * tangent).sum(axis=1) / (tangent * tangent).sum(axis=1)
    return t
