import hashlib
import pandas as pd
import numpy as np

# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.
def create_sample_split(df: pd.Dataframe, id_column: str, training_frac: float = 0.8) -> pd.Dataframe:
    """Create sample split based on ID column.

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

    Notes
    -----
    A hexadecimal string is a base-16 representation of data.
	UTF-8 is a standard text encoding that can represent any character using one to four bytes.
	hashlib.sha256 is a function used to compute the SHA-256 hash of a given input, which is useful for creating unique, secure identifiers or checksums.
    """
    # Create a unique key by concatenating the specified columns as a string
    def generate_hash(row):
        combined_key = '_'.join(str(row[col]) for col in id_column)
        # Create a hash of the combined key
        return int(hashlib.sha256(combined_key.encode('utf-8')).hexdigest(), 16) % 100

    # Apply the hash function to each row and assign it to a new column 'hash_group'
    df['hash_group'] = df.apply(generate_hash, axis=1)
    
    # Determine the threshold for the training split
    train_threshold = training_frac * 100
    
    # Create the 'sample' column based on the hash group
    df['sample'] = df['hash_group'].apply(lambda x: 'train' if x < train_threshold else 'test')
    
    # Drop the 'hash_group' column as it's only needed for internal calculation
    df.drop(columns=['hash_group'], inplace=True)
    
    return df


########## Without hashlib ##########
# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.
def split_nohash(df: pd.Dataframe, id_column: str, training_frac: float = 0.8) -> pd.Dataframe:
    """Create sample split based on ID column.

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
   # Create a unique identifier by concatenating the key columns
    combined_key = df[id_column].apply(lambda row: '_'.join(str(x) for x in row), axis=1)
    
    # Convert the combined key to an integer sum of ASCII values
    ascii_sum = combined_key.apply(lambda x: sum(ord(char) for char in x))
    
    # Apply modulo 100 to distribute into 100 buckets
    df['bucket'] = ascii_sum % 100
    
    # Determine the threshold for the training split
    train_threshold = training_frac * 100
    
    # Assign 'train' or 'test' based on the bucket
    df['sample'] = df['bucket'].apply(lambda x: 'train' if x < train_threshold else 'test')
    
    # Drop the 'bucket' column if not needed for output
    df.drop(columns=['bucket'], inplace=True)
    
    return df