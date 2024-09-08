import dask.dataframe as dd
from dask.dataframe.core import DataFrame as DaskDataFrame 
from typing import List
    
def load_lazy_df(path: str, usecols: List[str], column_names: List[str])->DaskDataFrame:
    """Load a dataframe in lazy fashion using dask."""
    filetype = get_filetype(path)
    if filetype == 'csv':
        df = load_df_from_csv(path, usecols)
    elif filetype == 'parquet':
        df = load_df_from_parquet(path, usecols)
    else:
        raise ValueError(f"Filetype {filetype} not supported")
    return df.rename(columns = {old: new for old, new in zip(usecols, column_names)})

def load_df_from_csv(path: str, usecols: List[str])->DaskDataFrame:
    """Load from csv using dask. usecols: columns to load."""    
    return dd.read_csv(path, usecols = usecols)

def load_df_from_parquet(path: str, usecols: List[str])->DaskDataFrame:
    """Load from parquet using dask. usecols: columns to load."""
    try:
        df = dd.read_parquet(path, columns = usecols)
    except Exception as e:
        df = dd.read_parquet(path)
        df = df[usecols]


    return df

def get_filetype(path: str)->str:
    """Get the filetype from a path."""
    return path.split('.')[-1]

def create_chunks(patients: list, chunk_size: int)->List[List]:
    """Given a list-like object, create chunks of patients."""
    return [patients[i:i+chunk_size] for i in range(0, len(patients), chunk_size)]
