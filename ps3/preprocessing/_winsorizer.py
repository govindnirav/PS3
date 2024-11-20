import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# TODO: Write a simple Winsorizer transformer which takes a lower and upper quantile and cuts the
# data accordingly
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile : float = 0.05, upper_quantile : float = 0.95):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None): # y is meant to be target variable in surpervised learning
        X=pd.DataFrame(X)
        quantiles = {}
        for i in X.columns:
            if pd.api.types.is_numeric_dtype(X[i]): # For each numeric column, it calculates the upper and lower quantiles and adds them to the respective arrays
                lower_quantile_ = X[i].quantile(self.lower_quantile)
                upper_quantile_ = X[i].quantile(self.upper_quantile)
                quantiles[i] = (lower_quantile_, upper_quantile_)
        self.quantiles_ = quantiles
        return self

    def transform(self, X):
        X=pd.DataFrame(X)
        for i in X.columns:
            if i in self.quantiles_:
                X[i] = X[i].clip(self.quantiles_[i][0], self.quantiles_[i][1])
        return X 
    
    def set_output(self, transform='default'):
        # This method is required to support the set_output method in the pipeline
        return self