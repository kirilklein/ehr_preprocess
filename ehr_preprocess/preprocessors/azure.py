import dask.dataframe as dd
import pandas as pd
from os.path import join
import os
from preprocessors.load import get_datastore
from azureml.core import Dataset


class AzurePreprocessor():
    # load data in dask
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.datastore, self.dump_path =  get_datastore()
        self.test = cfg.test

    def __call__(self):
        self.patients_info()
        self.concepts()

    def concepts(self):
        # Loop over all top-level concepts (diagnosis, medication, procedures, etc.)
        admissions = self.get_admissions()
        for type, top_level_config in self.cfg.concepts.items():
            print(f"INFO: Preprocess {type}")
            concepts = self.load_dask(top_level_config)
            concepts = self.select_columns(concepts, top_level_config)
            concepts = concepts.dropna()
            concepts = concepts.drop_duplicates()
            if type=='medication':
                ord['CONCEPT'] = ord['CONCEPT'].map(lambda x: 'M'+x)
                concepts['CONCEPT'] = concepts['CONCEPT'].map(lambda x: 'M'+x)
            concepts = self.change_dtype(concepts, top_level_config)
            concepts = concepts.set_index('PID')
            concepts = concepts.repartition(top_level_config.npartitions)
            concepts = self.assign_segment(concepts, admissions)
            self.save(concepts.compute(), f'concept.{type}')

    def patients_info(self):
        print("Load patients info")
        df = self.load_pandas(self.cfg.patients_info)
        if self.test:
            df = df.sample(10000)
        df = self.select_columns(df, self.cfg.patients_info)
        # Convert info dict to dataframe
        self.save(df, 'patients_info')

    def assign_segment(self, concepts: dd.DataFrame, admissions: dd.DataFrame)->dd.DataFrame:
        merged = concepts.merge(admissions, how='inner', left_index=True, right_index=True, suffixes=('', '_ADMISSION'))
        merged = merged[(merged['TIMESTAMP'] >= merged['TIMESTAMP_ADMISSION']) & (merged['TIMESTAMP'] <= merged['TIMESTAMP_END'])]
        merged = merged.drop(columns=['TIMESTAMP_ADMISSION', 'TIMESTAMP_END'])
        return merged

    def get_admissions(self):
        print("Load admissions")
        df = self.load_pandas(self.cfg.admissions)
        df = self.select_columns(df, self.cfg.admissions)
        df['SEGMENT'] = df.groupby('PID').cumcount()+1
        df = df.set_index('PID')
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
        ds = self.get_dataset(cfg)
        df = ds.to_pandas_dataframe()
        return df
    
    def load_dask(self, cfg: dict):
        """Change in azure to"""
        ds = self.get_dataset(cfg)
        df = ds.to_dask_dataframe()
        return df

    def get_dataset(self, cfg: dict):
        ds = Dataset.Tabular.from_parquet_files(path=(self.datastore, join(self.dump_path, cfg.filename)))
        if self.test:
            ds = ds.take(10000)
        return ds
    
    def save(self, df, filename):
        print(f"Save {filename}")
        out = self.cfg.paths.output_dir
        if not os.path.exists(out):
            os.makedirs(out)
        if self.cfg.paths.file_type == 'parquet':
            path = os.path.join(out, f'{filename}.parquet')
            df.to_parquet(path)
        elif self.cfg.paths.file_type == 'csv':
            path = os.path.join(out, f'{filename}.csv')
            df.to_csv(path, index=False)
