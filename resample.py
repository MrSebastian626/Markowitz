import pandas as pd
import numpy as np

def resampled_efficient_frontier(
    model_class,
    returns: pd.DataFrame,
    n_samples: int = 5,
    top_n: int = 200,
    target_return: float = 0.15,
    delta_fraction: float = 0.3
):
    """
    Run resampled efficient frontier by bootstrapping returns and averaging weights.

    Parameters:
        model_class: A class (like RobustMarkowitz) that takes returns and solves an optimization.
        returns (pd.DataFrame): Historical return data (e.g., past 20 years).
        n_samples (int): Number of bootstrapped resamples to run.
        top_n (int): Number of assets to retain based on Sharpe ratio.
        target_return (float): Target return for optimization.
        delta_fraction (float): Uncertainty level for box robust optimization.
        **model_kwargs: Additional parameters passed to model_class (e.g., weights=...).

    Returns:
        avg_weights (pd.Series): Averaged weights across simulations.
        cumulative_returns (pd.Series): Cumulative return curve from applying avg_weights.
    """
    weights_list = []

    for i in range(n_samples):
        try:
            # Bootstrap sample of historical returns
            sampled_returns = block_bootstrap(returns, block_size=21)
            # Instantiate model and solve
            model = model_class(sampled_returns, top_n=top_n)
            weights, _ = model.solve_box_uncertainty(target_return=target_return, delta_fraction=delta_fraction)
            weights_list.append(weights)
        except Exception as e:
            print(f"⚠️ Iteration {i+1} failed: {e}")

    if not weights_list:
        raise ValueError("❌ No successful optimizations in resampling.")

    # Average weights across all successful samples
    avg_weights = pd.concat(weights_list, axis=1).mean(axis=1)
    avg_weights = avg_weights / avg_weights.sum()
    # Apply average weights to original return series
    aligned_returns = returns[avg_weights.index]
    port_returns = aligned_returns @ avg_weights
    cumulative_returns = (1 + port_returns).cumprod()
    return avg_weights, cumulative_returns

def block_bootstrap(returns: pd.DataFrame, block_size: int = 21) -> pd.DataFrame:
    """
    Perform block bootstrap sampling on time series returns.

    Parameters:
        returns (pd.DataFrame): Original returns with datetime index.
        block_size (int): Number of days per block (e.g., 21 trading days = ~1 month).

    Returns:
        pd.DataFrame: Resampled returns with preserved datetime index.
    """
    num_days = len(returns)
    num_blocks = num_days // block_size

    block_starts = np.random.randint(0, num_days - block_size, size=num_blocks)
    sampled = [returns.iloc[start:start + block_size] for start in block_starts]
    
    # Preserve datetime index
    return pd.concat(sampled)

