import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

def nearest_psd(A, epsilon=1e-8):
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals[eigvals < epsilon] = epsilon
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

class CvarRobustMarkowitz:
    def __init__(self, returns: pd.DataFrame, top_n: int = 200, weights: np.ndarray = None):
        if weights is None:
            weights = np.array([0.1, 0.2, 0.3, 0.4])
        assert len(weights) == 4, "weights must be a 4-element array"
        weights = weights / weights.sum()

        total_days = len(returns)
        block_size = total_days // 4
        blocks = [returns.iloc[i*block_size:(i+1)*block_size] for i in range(4)]

        annual_returns = []
        for block in blocks:
            block_years = (block.index[-1] - block.index[0]).days / 365.25
            compounded = (1 + block).prod() ** (1 / block_years) - 1
            annual_returns.append(compounded)

        weighted_mu = sum(w * r for w, r in zip(weights, annual_returns))

        mu_all = returns.mean() * 252
        vol_all = returns.std() * np.sqrt(252)
        sharpe_ratio = mu_all / vol_all
        top_assets = sharpe_ratio.sort_values(ascending=False).head(top_n).index

        self.returns = returns[top_assets]
        self.mu = weighted_mu[top_assets]
        raw_cov = LedoitWolf().fit(self.returns).covariance_ * 252
        self.Sigma = nearest_psd(raw_cov)
        self.n = len(self.mu)

    def solve_box_uncertainty(self, alpha=0.90, delta_fraction=0.2, target_return=0.10):
        mu_bar = self.mu.values
        delta = delta_fraction * np.abs(mu_bar)
        worst_case_mu = mu_bar - delta

        T = self.returns.shape[0]
        w = cp.Variable(self.n)
        z = cp.Variable(T)
        eta = cp.Variable()

        loss = -self.returns.values @ w
        constraints = [
            cp.sum(w) == 1,
            worst_case_mu @ w >= target_return,
            w >= 0,
            z >= 0,
            z >= loss - eta
        ]

        cvar_obj = eta + (1 / (1 - alpha)) * cp.sum(z) / T
        prob = cp.Problem(cp.Minimize(cvar_obj), constraints)

        try:
            prob.solve(verbose=False)
            if prob.status in ["optimal", "optimal_inaccurate"]:
                weights = pd.Series(w.value, index=self.returns.columns)
                port_rets = self.returns @ weights
                cumulative = (1 + port_rets).cumprod()
                return weights, cumulative
            else:
                raise ValueError(f"Solver failed: {prob.status}")
        except Exception as e:
            print("CVaR Optimization failed:", e)
            raise
