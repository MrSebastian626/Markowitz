# factor.py

import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
import cvxpy as cp
from robust import RobustMarkowitz

class FactorModelMarkowitz(RobustMarkowitz):
    def __init__(
        self,
        returns: pd.DataFrame,
        betas_path: str,
        macro_path: str,
        current_year: int,
        gamma: float = 1.0,
        top_n: int = 200,
        weights: np.ndarray = None
    ):
        """
        A robust Markowitz model adjusted with macroeconomic betas.

        Parameters:
            returns (pd.DataFrame): Daily return data over 20 years.
            betas_path (str): CSV file path with stock x macro beta values.
            macro_path (str): CSV file path of yearly macro indicators.
            current_year (int): Year of macro data to use (reads year-1 from macro).
            gamma (float): Strength of macro influence [0 = ignore].
            top_n (int): Number of top Sharpe ratio assets to retain.
            weights (np.ndarray): Optional weights for 4 x 5-year return blocks.
        """
        # Initialize base robust model (geometric return weighted)
        super().__init__(returns, top_n=top_n, weights=weights)

        # Load macro indicators for the previous year
        macro_df = pd.read_csv(macro_path, index_col="Year")
        macro_vector = macro_df.loc[current_year - 1]

        # Load beta estimates
        betas_df = pd.read_csv(betas_path, index_col=0)
        betas_df = betas_df.loc[self.returns.columns].fillna(0)

        # Adjust expected returns based on macro factors
        macro_impact = betas_df.values @ macro_vector.values
        adjusted_mu = self.mu + gamma * pd.Series(macro_impact, index=self.mu.index)

        self.mu = adjusted_mu
