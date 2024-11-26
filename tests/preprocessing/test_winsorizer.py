import numpy as np
import pytest

from ps3.preprocessing import Winsorizer

# TODO: Test your implementation of a simple Winsorizer
@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
#def test_winsorizer(lower_quantile, upper_quantile):

#    X = np.random.normal(0, 1, 1000)

#    assert False # This makes me fail everytime - I assume I had to define the test myself

def test_winsorizer_correct(lower_quantile, upper_quantile): # Corrected
    X = np.random.normal(0, 1, 1000)

    Xt = Winsorizer(lower_quantile, upper_quantile).fit_transform(X)

    assert (Xt.max() == np.percentile(X, upper_quantile * 100)) & (Xt.min() == np.percentile(X, lower_quantile * 100))
    # Assert just checks if the condition is true, if not it raises an error.

### TODO:Write a test for preprocessor.set_output(transform = "pandas") method yields a pandas DataFrame