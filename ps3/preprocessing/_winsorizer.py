import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# TODO: Write a simple Winsorizer transformer which takes a lower and upper quantile and cuts the
# data accordingly
class Winsorizer(BaseEstimator, TransformerMixin): # Base estimator is the base class for all estimators in scikit-learn.
    def __init__(self, lower_quantile : float = 0.05, upper_quantile : float = 0.95):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile


    def fit(self, X, y=None): # Corrected
        self.lower_bound = np.percentile(X, self.lower_quantile * 100)
        self.upper_bound = np.percentile(X, self.upper_quantile * 100)
        return self
    

    def transform(self, X): # Corrected
        check_is_fitted(self) # Check if the estimator is fitted by verifying the presence of fitted attributes (ending with a trailing underscore)
        X_clipped = np.clip(X, self.lower_bound, self.upper_bound)
        return X_clipped

    
    def set_output(self, transform='default'):
        # This method is required to support the set_output method in the pipeline
        return self 