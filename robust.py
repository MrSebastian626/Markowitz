import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

def nearest_psd(A, epsilon=1e-8):
    """
    Projects a matrix to the nearest positive semi-definite (PSD) matrix.
    
    Parameters:
        A (np.ndarray): Input matrix
        epsilon (float): Minimum allowed eigenvalue to ensure PSD
        
    Returns:
        np.ndarray: PSD matrix
    """
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals[eigvals < epsilon] = epsilon
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

class RobustMarkowitz:
    def __init__(self, returns: pd.DataFrame, top_n: int = 200, weights: np.ndarray = None):
        """
        Initialize RobustMarkowitz with weighted geometric mean returns.

        Parameters:
            returns (pd.DataFrame): Historical returns for 20 years.
            top_n (int): Number of top Sharpe ratio assets to keep.
            weights (np.ndarray): Optional weights for each 5-year block (length 4).
        """
        if weights is None:
            weights = np.array([0.1, 0.2, 0.3, 0.4])  # default emphasis on recent data
        assert len(weights) == 4, "weights must be a 4-element array"
        weights = weights / weights.sum()  # normalize

        # Step 1: Split returns into 4 equal-length blocks
        total_days = len(returns)
        block_size = total_days // 4
        blocks = [returns.iloc[i*block_size:(i+1)*block_size] for i in range(4)]

        # Step 2: Compute geometric annual return for each block
        annual_returns = []
        for block in blocks:
            block_years = (block.index[-1] - block.index[0]).days / 365.25
            compounded = (1 + block).prod() ** (1 / block_years) - 1
            annual_returns.append(compounded)

        # Step 3: Weighted geometric mean
        weighted_mu = sum(w * r for w, r in zip(weights, annual_returns))

        # Step 4: Filter top N by Sharpe ratio (based on full 20-year data)
        mu_all = returns.mean() * 252
        vol_all = returns.std() * np.sqrt(252)
        sharpe_ratio = mu_all / vol_all
        top_assets = sharpe_ratio.sort_values(ascending=False).head(top_n).index

        # Step 5: Store data
        self.returns = returns[top_assets]
        self.mu = weighted_mu[top_assets]
        raw_cov = LedoitWolf().fit(self.returns).covariance_ * 252
        self.Sigma = nearest_psd(raw_cov)  # <- Ensure covariance matrix is PSD
        self.n = len(self.mu)

    def solve_box_uncertainty(self, target_return=0.10, delta_fraction=0.2, min_target=0.05, step=0.01):
        mu_bar = self.mu.values
        delta = delta_fraction * np.abs(mu_bar)
        w = cp.Variable(self.n)

        while target_return >= min_target:
            worst_case_return = mu_bar @ w - delta @ w
            constraints = [
                cp.sum(w) == 1,
                worst_case_return >= target_return,
                w >= 0
            ]
            prob = cp.Problem(cp.Minimize(cp.quad_form(w, self.Sigma)), constraints)

            try:
                prob.solve(verbose=False)
                if prob.status in ["optimal", "optimal_inaccurate"]:
                    weights = pd.Series(w.value, index=self.returns.columns)
                    port_rets = self.returns @ weights
                    cumulative = (1 + port_rets).cumprod()
                    return weights, cumulative
                else:
                    print(f"⚠️ Infeasible for target return {target_return:.2%}. Trying lower...")
            except Exception as e:
                print(f"⚠️ Solver failed for target return {target_return:.2%}: {e}")

            target_return -= step

        raise ValueError("❌ No feasible solution found for box uncertainty.")

    def solve_ellipsoidal_uncertainty(self, target_return=0.10, rho=0.05, min_target=0.05, step=0.01):
        mu_bar = self.mu.values
        w = cp.Variable(self.n)

        while target_return >= min_target:
            constraints = [
                cp.sum(w) == 1,
                mu_bar @ w - rho * cp.norm(w, 2) >= target_return,
                w >= 0
            ]
            prob = cp.Problem(cp.Minimize(cp.quad_form(w, self.Sigma)), constraints)

            try:
                prob.solve(verbose=False)
                if prob.status in ["optimal", "optimal_inaccurate"]:
                    weights = pd.Series(w.value, index=self.returns.columns)
                    port_rets = self.returns @ weights
                    cumulative = (1 + port_rets).cumprod()
                    return weights, cumulative
                else:
                    print(f"⚠️ Infeasible for target return {target_return:.2%}. Trying lower...")
            except Exception as e:
                print(f"⚠️ Solver failed for target return {target_return:.2%}: {e}")

            target_return -= step

        raise ValueError("❌ No feasible solution found for ellipsoidal uncertainty.")
