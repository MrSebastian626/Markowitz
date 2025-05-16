import numpy as np
import pandas as pd
import cvxpy as cp

def nearest_psd(A, epsilon=1e-8):
    """
    Projects a matrix to the nearest positive semi-definite (PSD) matrix.
    """
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals[eigvals < epsilon] = epsilon
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def classical_markowitz(returns, target_return=0.15):
    """
    Classical mean-variance optimization using all available assets.
    Uses compounded growth for expected returns and sample covariance.

    Parameters:
        returns (pd.DataFrame): T x N matrix of returns
        target_return (float): Minimum expected annual return

    Returns:
        weights (pd.Series), cumulative_returns (pd.Series)
    """
    # Step 1: Compute expected returns (annualized)
    mu = returns.mean() * 252

    # Step 2: Sample covariance matrix (annualized)
    cov = returns.cov() * 252
    cov = nearest_psd(cov)  # ensure it's PSD
    cov = pd.DataFrame(cov, index=returns.columns, columns=returns.columns)

    # Step 3: Filter out problematic assets
    valid_assets = mu.index[~(mu.isna() | np.isinf(mu))]
    mu = mu.loc[valid_assets]
    cov = cov.loc[valid_assets, valid_assets]
    returns = returns[valid_assets]
    n = len(mu)

    # Step 4: Convex optimization
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cov))
    constraints = [
        cp.sum(w) == 1,
        mu.values @ w >= target_return,
        w >= 0,
        w <= 1,
    ]

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(verbose=False)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Solver status: {prob.status}")

        weights = pd.Series(w.value, index=returns.columns)
        portfolio_returns = returns @ weights
        cumulative_returns = (1 + portfolio_returns).cumprod()

        return weights, cumulative_returns

    except Exception as e:
        print("Optimization failed:", e)
        raise
