"""Kalman filters for the unmeasured core temperature (RQ1).

KF  (augment=False): state [Tc, Ts], heat input from measured current and nominal R_int.
AKF (augment=True):  state [Tc, Ts, theta], with Q = theta * I^2 * R_int,nominal and theta
                     a slow random walk. This lets the filter detect and correct
                     "the cell makes more heat than the model thinks". The model stays
                     linear in the state, so it is still an ordinary (time-varying) KF.

Both filters use the nominal cell parameters, whatever the true plant is.
"""

from __future__ import annotations

import numpy as np

from thermal_env import NOMINAL, SENSOR_SIGMA, CellParams, discretise


class KalmanFilter:
    def __init__(self, augment: bool = True, params: CellParams = NOMINAL,
                 q_T: float = 0.05, q_theta: float = 0.003, r: float = SENSOR_SIGMA**2,
                 P0_T: float = 2.0, P0_theta_var: float = 0.1):
        self.augment = augment
        self.params = params
        self.n = 3 if augment else 2
        self.r = r
        self.Qk = np.diag([q_T**2, q_T**2, q_theta**2][: self.n])
        self._P0 = np.diag([P0_T**2, P0_T**2, P0_theta_var][: self.n])
        self.reset()

    @property
    def name(self) -> str:
        return "AKF" if self.augment else "KF"

    def reset(self, T0: float | None = None) -> None:
        T0 = self.params.Tf if T0 is None else T0
        self.x = np.array([T0, T0, 1.0][: self.n])
        self.P = self._P0.copy()

    # -- model ---------------------------------------------------------------
    def model(self, I, u):
        """Transition x' = F x + c for current I and cooling u. Both may be scalars or arrays
        (broadcast together), e.g. a different current for each future step."""
        u, I = np.asarray(u, dtype=float), np.asarray(I, dtype=float)
        shape = np.broadcast_shapes(u.shape, I.shape)
        Ad, Bd = discretise(self.params, np.broadcast_to(u, shape))
        q = np.broadcast_to(I * I * self.params.R_int, shape)[..., None]
        F = np.zeros(shape + (self.n, self.n))
        c = np.zeros(shape + (self.n,))
        F[..., :2, :2] = Ad
        if self.augment:
            F[..., :2, 2] = Bd[..., :, 0] * q
            F[..., 2, 2] = 1.0
            c[..., :2] = Bd[..., :, 1] * self.params.Tf
        else:
            c[..., :2] = Bd[..., :, 0] * q + Bd[..., :, 1] * self.params.Tf
        return F, c

    # -- filter steps ------------------------------------------------------------
    def predict(self, I: float, u: float) -> None:
        F, c = self.model(I, u)
        self.x = F @ self.x + c
        self.P = F @ self.P @ F.T + self.Qk

    def update(self, z: float) -> tuple[float, float]:
        """Measurement update with surface reading z. Returns (innovation, its variance)."""
        nu = z - self.x[1]
        S = self.P[1, 1] + self.r
        K = self.P[:, 1] / S
        self.x = self.x + K * nu
        IKH = np.eye(self.n) - np.outer(K, np.eye(self.n)[1])
        self.P = IKH @ self.P @ IKH.T + self.r * np.outer(K, K)  # Joseph form
        return float(nu), float(S)

    # -- outputs -------------------------------------------------------------------
    @property
    def Tc_hat(self) -> float:
        return float(self.x[0])

    @property
    def sigma_Tc(self) -> float:
        return float(np.sqrt(self.P[0, 0]))

    @property
    def theta_hat(self) -> float:
        return float(self.x[2]) if self.augment else float("nan")

    def forecast(self, I, u_seq, return_final: bool = False):
        """Open-loop prediction (no measurement updates) for a batch of input sequences.

        u_seq: array (..., H), the cooling command for each future step. I: the current, either
        a scalar (held constant) or a per-step array broadcastable to u_seq (a load preview).
        Returns (Tc mean, Tc sigma), each (..., H), for steps 1..H; with return_final=True
        also the full mean and covariance after the last step.
        """
        u_seq = np.asarray(u_seq, dtype=float)
        F, c = self.model(I, u_seq)  # (..., H, n, n), (..., H, n)
        H = F.shape[-3]
        batch = F.shape[:-3]
        m = np.broadcast_to(self.x, batch + (self.n,)).copy()
        P = np.broadcast_to(self.P, batch + (self.n, self.n)).copy()
        mean = np.empty(batch + (H,))
        sig = np.empty(batch + (H,))
        for t in range(H):
            Ft = F[..., t, :, :]
            m = np.einsum("...ij,...j->...i", Ft, m) + c[..., t, :]
            P = Ft @ P @ np.swapaxes(Ft, -1, -2) + self.Qk
            mean[..., t] = m[..., 0]
            sig[..., t] = np.sqrt(P[..., 0, 0])
        return (mean, sig, m, P) if return_final else (mean, sig)
