import pandas as pd
def filter_df_by_patients(df: pd.DataFrame, patients: list, pid_col: str='PID')->pd.DataFrame:
    """Filter the dataframe by patients."""
    return df[df[pid_col].isin(patients)]