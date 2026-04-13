"""
Configuration Module for Friend_Eb_Project Architecture
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Dynamic Architecture Constraints
NUM_SAMPLES = 720
NUM_STRUCTURAL = 40
NUM_PHYSICAL = 40
NUM_POLYBERT_PCA = 40
NUM_MORGAN_PCA = 40

# Optimization Bounds
TEMP_BOUNDS = (100, 300)
CRYST_BOUNDS = (0.10, 0.90)

TARGET_TG_NOISE = 3.5
RANDOM_SEED = 42

# ----------------- Custom Transformers ----------------- #
class CollinearityDropper(BaseEstimator, TransformerMixin):
    """
    Safely calculates and drops extremely highly correlated features natively within an sklearn Pipeline
    fitted strictly upon X_train space.
    """
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.drop_columns_ = []
        
    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        corr_matrix = df.corr().abs().fillna(0)
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # Map indices to drop
        self.drop_columns_ = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return self
        
    def transform(self, X, y=None):
        df = pd.DataFrame(X)
        df_dropped = df.drop(columns=self.drop_columns_)
        return df_dropped.values
