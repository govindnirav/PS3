import hashlib
import pandas as pd
import numpy as np

# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.
def create_sample_split(df: pd.DataFrame, id_column: str, training_frac: float = 0.8) -> pd.DataFrame:
    """
    Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.8

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """

    if df[id_column].dtype == np.int64: # If the column is an integer, we can simply take the modulo.
        modulo = df[id_column] % 100
    else: # If the column is not an integer, we need to hash it. Hashing is a process of converting an input (like a string) into a fixed-size string of bytes.
        modulo = df[id_column].apply(
            lambda x: int(hashlib.md5(str(x).encode()).hexdigest(), 16) % 100
        )
        """
            md5 is a widely used cryptographic hash function that produces a 128-bit hash value. Base 16 is used to convert the hash value to a hexadecimal string.
            Another popular hash function is SHA-256 which produces a 256-bit hash value.
            Rows that are close together in small changes will have very different hash values, so could end up on different sides of the split.
            Computing the modulo of the hash value by 100 will give us a number between 0 and 99. It will also yield the same results for the same input on any machine.
        """
    
    df["sample"] = np.where(modulo < training_frac * 100, "train", "test") # All modulos below 80 are train, the rest are test

    return df
