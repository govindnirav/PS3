import hashlib
import pandas as pd
import numpy as np

# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.
def create_sample_split(df, id_column, training_frac = 0.8):
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

#def correct_split_hash(df: pd.Dataframe, id_column: str, training_frac: float = 0.8) -> pd.Dataframe:
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
    
    df['sample'] = np.where(modulo < training_frac * 100, 'train', 'test') # All modulos below 80 are train, the rest are test

    return df


########## Without hashlib ##########
# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.

#def split_nohash(df: pd.Dataframe, id_column: str, training_frac: float = 0.8) -> pd.Dataframe:
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

