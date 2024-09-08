import json
import os
from os.path import dirname, join, realpath, split

import pandas as pd
from hydra.utils import instantiate

base_dir = dirname(dirname(dirname(realpath(__file__))))

class BasePreprocessor():
    def __init__(self, 
            cfg, 
        ):
        self.config = cfg
        self.set_test(cfg)
        self.info = {}
        self.admission_info = {}

        if not os.path.exists(self.config.paths.output_dir):
            os.makedirs(self.config.paths.output_dir)
            
    def set_test(self, cfg):
        if 'test' in cfg:
            self.test = cfg.test
        else:
            self.test = False

    def __call__(self):
        self.process()

    # All saving is handled internally in the functions below
    def process(self):
        self.concepts()
        self.patients_info()

    def concepts(self):
        # Loop over all top-level concepts (diagnosis, medication, procedures, etc.)
        for type, top_level_config in self.config.concepts.items():
            individual_dfs = [self.load_csv(cfg) for cfg in top_level_config.values()]
            combined = pd.concat(individual_dfs)
            combined = combined.drop_duplicates(subset=['PID', 'CONCEPT', 'TIMESTAMP'])
            combined = combined.sort_values('TIMESTAMP')

            self.save(combined, f'concept.{type}')

    def patients_info(self):
        for key, cfg in self.config.patients_info.items():
            df = self.load_csv(cfg)
            
            self.set_patient_info(df)

        # Convert info dict to dataframe
        df = self.info_to_df()

        self.save(df, 'patients_info')

    def set_patient_info(self, df: pd.DataFrame):
        df = df.set_index('pid_col')
        df.apply(lambda patient:
            self.info.get(patient.name)             # patient.name is the index PID
            .update(
                {k: v for k,v in patient.items()}   # Add every column to patient info
            ), axis=1
        )

    def add_patients_to_info(self, df: pd.DataFrame):
        for pat in df['PID'].unique():
            self.info.setdefault(pat, {})

    def add_admission_info(self, df: pd.DataFrame):
        for _, subset in df.groupby('ADMISSION_ID'):
            # Remove rows with missing values
            subset = subset[subset[['ADMISSION_ID', 'PID', 'TIMESTAMP']].notna().all(1)]
            if len(subset) == 0:
                continue

            # Get first chronological row of subset
            subset = subset.sort_values('TIMESTAMP')
            adm = subset['ADMISSION_ID'].iloc[0]
            pid = subset['PID'].iloc[0]
            timestamp = subset['TIMESTAMP'].iloc[0]

            # Add admission info to admission dict
            adm_dict = self.admission_info.setdefault(adm, {})
            adm_dict.update(
                PID=pid,
                TIMESTAMP=timestamp
            )

    def info_to_df(self, pid_col: str='PID'):
        df = pd.DataFrame.from_dict(self.info, orient="index")
        df = df.reset_index().rename(columns={'index': pid_col})    # Convert the PID index to PID column
        return df

    def load_csv(self, cfg: dict):
        converters = self.get_converters(cfg)
        parse_dates = self.get_parse_dates(cfg)
        df = self.read_csv_file(cfg, converters, parse_dates)
        self.add_info_to_dicts(df)
        df = self.apply_function(cfg, df)
        return df
    
    def get_dtypes(self, cfg: dict):
        if cfg.get('dtype') is not None:
            dtypes = {column: dtype for column, dtype in cfg.get('dtype').items()}
        else: 
            dtypes = None
        return dtypes

    def get_converters(self, cfg: dict):
        if cfg.get('converters') is not None:
            converters = {column: instantiate(func) for column, func in cfg.get('converters').items()}
        else: 
            converters = None
        return converters

    def get_parse_dates(self, cfg: dict):
        if cfg.get('parse_dates') is not None:
            parse_dates = [column for column in cfg.get('parse_dates')]
        else: 
            parse_dates = None
        return parse_dates
    
    def read_csv_file(self, cfg: dict, converters: dict, parse_dates: list):
        df = pd.read_csv(
            join(self.config.paths.main_folder, cfg['filename']),
            converters=converters,
            usecols=cfg.get('usecols'),
            names=cfg.get('names'),
            dtype=self.get_dtypes(cfg),
            parse_dates=parse_dates,
            encoding='ISO-8859-1',
            skiprows=[0],
            header=0,
            nrows=10000 if self.test else None
        )
        return df

    def add_info_to_dicts(self, df):
        if 'PID' in df.columns:
            self.add_patients_to_info(df)
            if 'ADMISSION_ID' in df.columns and 'TIMESTAMP' in df.columns:
                self.add_admission_info(df)

    def apply_function(self, cfg: dict, df):
        if cfg.get('function') is not None:
            df = instantiate(cfg['function'])(self, df)
        return df

    def save(self, df: pd.DataFrame, filename: str):
        if self.config.paths.file_type == 'parquet':
            path = os.path.join(self.config.paths.output_dir, f'{filename}.parquet')
            df.to_parquet(path)
        elif self.config.paths.file_type == 'csv':
            path = os.path.join(self.config.paths.output_dir, f'{filename}.csv')
            df.to_csv(path, index=False)


class BaseMIMICPreprocessor():
    def __init__(self, cfg, test=False) -> None:
        self.test = test
        self.cfg = cfg
        self.raw_data_path = cfg.paths.raw_data_path
        self.prepends = cfg.prepends
        data_folder_name = split(self.raw_data_path)[-1]

        if not cfg.paths.working_data_path is None:
            working_data_path = cfg.paths.working_data_path
        else:
            working_data_path = join(base_dir, 'data')

        self.interim_data_path = join(
            working_data_path, 'interim', data_folder_name)
        if not os.path.exists(self.interim_data_path):
            os.makedirs(self.interim_data_path)
        self.formatted_data_path = os.getcwd()
        if not os.path.exists(self.formatted_data_path):
            os.makedirs(self.formatted_data_path)
        if test:
            self.nrows = 10000
        else:
            self.nrows = None
        if os.path.exists(join(self.formatted_data_path, 'metadata.json')):
            with open(join(self.formatted_data_path, 'metadata.json'), 'r') as fp:
                self.metadata_dic = json.load(fp)
        else:
            self.metadata_dic = {}

    def save_metadata(self):
        with open(join(self.formatted_data_path, 'metadata.json'), 'w') as fp:
            json.dump(self.metadata_dic, fp)

    def update_metadata(self, concept_name, coding_system, src_files_ls):
        if concept_name!='patients_info':
            file = f'concept.{concept_name}.parquet'
            prepend = self.prepends[concept_name]
        else:
            file = 'patients_info.parquet'
            prepend = None
        concept_dic = {
            'Coding_System': coding_system, 'Prepend': prepend, 'Source': src_files_ls
        }
        if file not in self.metadata_dic:
            self.metadata_dic[file] = concept_dic
    


    