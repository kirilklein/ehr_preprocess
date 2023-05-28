from preprocessors.base import BasePreprocessor
import dask.dataframe as dd
import pandas as pd
from os.path import join

from azureml.core import Workspace, ScriptRunConfig, Environment, Experiment
from azureml.core.runconfig import MpiConfiguration

# get workspace
ws = Workspace.from_config()

# get compute target
target = ws.compute_targets['Kiril-CPU']

# get curated environment
curated_env_name = 'AzureML-PyTorch-1.6-GPU'
env = Environment.get(workspace=ws, name=curated_env_name)

from azureml.core import Workspace, Dataset, Datastore
from os.path import join

subscription_id = 'f8c5aac3-29fc-4387-858a-1f61722fb57a'
resource_group = 'forskerpl-n0ybkr-rg'
workspace_name = 'forskerpl-n0ybkr-mlw'
  
workspace = Workspace(subscription_id, resource_group, workspace_name)
datastore = Datastore.get(workspace, "researcher_data")
dump_path = join("data-backup", "SP-dumps", "2022-10-27")


class AzurePrepocessor(BasePreprocessor):
    # load data in dask
    def __init__(self, cfg, test=False) -> None:
        super().__init__(cfg, test)
        self.batch_size = self.cfg.preprocessor.batch_size

    def concepts(self):
        # Loop over all top-level concepts (diagnosis, medication, procedures, etc.)
        admissions = self.get_admissions(self.config.admissions)
        for type, top_level_config in self.config.concepts.items():
            concepts = self.load_dask(top_level_config)
            concepts = self.select_columns(concepts, top_level_config)
            concepts = concepts.dropna()
            concepts = concepts.drop_duplicates()
            concepts = self.change_dtype(concepts, top_level_config)
            concepts = self.assign_admission_id(concepts, admissions)
            self.save(concepts, f'concept.{type}')

    def patients_info(self):
        df = self.load_csv(self.config.patients_info)
        # Convert info dict to dataframe
        self.save(df, 'patients_info')

    def assign_admission_id(self, concepts: dd.DataFrame, admissions: dd.DataFrame)->dd.DataFrame:
        merged = concepts.merge(admissions, how='inner', left_index=True, right_index=True, suffixes=('', '_ADMISSION'))
        merged = merged[(merged['TIMESTAMP'] >= merged['TIMESTAMP_ADMISSION']) & (merged['TIMESTAMP'] <= merged['TIMESTAMP_END'])]
        merged = merged.drop(columns=['TIMESTAMP_ADMISSION', 'TIMESTAMP_END'])
        return merged

    def get_admissions(self):
        df = self.load_pandas(self.config.admissions)
        df = self.select_columns(df, self.config.admissions)
        df['SEGMENT'] = df['SEGMENT'] = df.groupby('PID').cumcount()+1
        return df
    
    def change_dtype(self, df, cfg):
        """Change column type"""
        for col, dtype in cfg.dtypes.items():
            df[col] = df[col].astype(dtype)
        return df

    def select_columns(self, df, cfg):
        """Select and Rename columns"""
        columns = df.columns.tolist()
        selected_columns = [columns[i] for i in cfg.usecols]
        df = df[selected_columns]
        df = df.rename(columns={old: new for old, new in zip(selected_columns, cfg.names)})
        return df
        
    def load_pandas(self, cfg: dict):
        ds = Dataset.Tabular.from_parquet_files(path=(datastore, join(dump_path, cfg.file_name)))
        df = ds.to_pandas_dataframe()
        return df
    
    def load_dask(cfg):
        """Change in azure to"""
        ds = Dataset.Tabular.from_parquet_files(path=(datastore, join(dump_path, cfg.file_name)))
        df = ds.to_dask_dataframe()
        return df