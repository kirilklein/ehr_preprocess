"""
Transforming the synthea data into the MEDS format as specified by:
https://github.com/Medical-Event-Data-Standard/meds
"""

import os
from os.path import join

import pandas as pd
import polars as pl
from MEDS_transforms.extract.finalize_MEDS_data import \
    get_and_validate_data_schema

from ehr_preprocess.preprocessors.base import BasePreprocessor
from ehr_preprocess.preprocessors.filter import filter_df_by_patients
from ehr_preprocess.preprocessors.loader import create_chunks, load_lazy_df
from ehr_preprocess.preprocessors.meds_transforms import (
    sort_final_dataframe, transform_admission_discharge,
    transform_patient_info)


class MEDSPreprocessor(BasePreprocessor):
    def __init__(self,  cfg: dict,chunk_size: int = 1000):
        super().__init__(cfg)
        self.chunk_size = chunk_size
        self.pid = cfg.pid
        self.time = cfg.time
        self.code = cfg.code
        self.dob = cfg.dob
        self.dod = cfg.dod
        self.admission = cfg.admission
        self.numerical_value = cfg.numerical_value
        self.static_vars = self.get_static_vars(cfg)
    
    def get_static_vars(self, cfg):
        """Get the static variables from the config."""
        static_vars = [
            name for key, value in cfg.items() if key == 'patients_info'
            for v in value.values()
            for name in v.get('names', [])
            if name not in {self.pid, self.dob, self.dod}
        ]
        return static_vars

    def __call__(self):
        patients = self.get_patient_ids()
        for i, pids_chunk in enumerate(create_chunks(patients, self.chunk_size)):
            result = self.process(pids_chunk)
            result = self.ensure_datatype_pd(result)
 
            #result = pl.from_pandas(result, schema_overrides={self.pid: pl.Int64(), self.time: pl.Datetime(), self.code: pl.String(), self.numerical_value: pl.Float64()}).lazy()
            #result = get_and_validate_data_schema(result, {'do_retype': True})
            self.save(result, f"meds_{i}.parquet")

    def ensure_datatype_pd(self, df):
        # Ensure the correct data types in the Pandas DataFrame
        df[self.pid] = df[self.pid].astype(pd.Int64Dtype())
        df[self.time] = pd.to_datetime(df[self.time], format='mixed')
        df[self.code] = df[self.code].astype(str)
        df[self.numerical_value] = df[self.numerical_value].astype(pd.Float64Dtype())
        return df

    def save(self, df: pl.DataFrame, filename: str):
        path = os.path.join(self.config.paths.output_dir, filename)
        df.to_parquet(path)

    def process(self, pids_chunk: list):
        """Process a chunk of patients, concatenating and sorting the results"""
        patients_info = self.patients_info(pids_chunk)
        concepts = self.concepts(pids_chunk)
        # concat and sort (first PID, then TIMESTAMP)
        result = pd.concat([patients_info, concepts], ignore_index=True)
        result = self.map_pids_to_int(result)
        result = sort_final_dataframe(result, self.pid, self.time, self.code, self.dob, self.dod, self.static_vars)

        return result        
    
    def map_pids_to_int(self, df: pd.DataFrame)->pd.DataFrame:
        """Map the patient ids to integers."""
        df[self.pid] = df[self.pid].apply(hash)
        return df

    def concepts(self, patients)->pd.DataFrame:
        """Loop over all top-level concepts (diagnosis, medication, procedures, etc.) and create one df"""
        all_dfs = []
        for type, top_level_config in self.config.concepts.items():
            individual_dfs = [self.load_df_for_patients(cfg, patients) for cfg in top_level_config.values()]
            combined = pd.concat(individual_dfs)
            combined = combined.drop_duplicates(subset=[self.pid, self.code, self.time])
            combined = transform_admission_discharge(combined, self.pid, self.admission, self.time, self.code)
            all_dfs.append(combined)
        result = pd.concat(all_dfs)
        return result
    
    def patients_info(self, patients: list)->pd.DataFrame:
        """Load and process the patient info dataframes."""
        for key, cfg in self.config.patients_info.items():
            df = self.load_df_for_patients(cfg, patients)
            self.set_patient_info(df)

        # Convert info dict to dataframe
        df = self.info_to_df()
        df = transform_patient_info(df, self.pid, self.dob, self.dod, self.time, self.code)
        return df

    def info_to_df(self)->pd.DataFrame:
        return pd.DataFrame.from_dict(self.info, orient="index")

    def set_patient_info(self, df: pd.DataFrame)->None:
        for pid, patient in df.iterrows():
            self.info.setdefault(pid, {}).update(patient.to_dict())

    def get_patient_ids(self)->pd.Series:  
        """Get the patient ids from the pids file."""
        path = join(self.config.paths.main_folder, self.config.pids['filename'])
        pid_col = self.config.pids.get('col')
        df = load_lazy_df(path, [pid_col], [self.pid])
        return df[self.pid].unique().compute()

    def load_df_for_patients(self, cfg: dict, patients: list)->pd.DataFrame:
        """Load a dataframe for a subset of patients."""
        path = join(self.config.paths.main_folder, cfg['filename'])
        df = load_lazy_df(path, cfg.get('usecols'), cfg.get('names'))
        return filter_df_by_patients(df, patients, self.pid).compute()
    
    
    



