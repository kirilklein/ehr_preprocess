import pandas as pd
from hydra.utils import instantiate


class BasePreprocessor():
    def __init__(self, 
            config, 
        ):
        self.config = config
        self.info = {}
        self.admission_info = {}

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
        df = df.set_index('PID')
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

    def info_to_df(self):
        df = pd.DataFrame.from_dict(self.info, orient="index")
        df = df.reset_index().rename(columns={'index': 'PID'})    # Convert the PID index to PID column
        return df

    def load_csv(self, cfg: dict):
        if cfg.get('converters') is not None:
            converters = {column: instantiate(func) for column, func in cfg.get('converters').items()}
        else: 
            converters = None
            
        if cfg.get('parse_dates') is not None:
            parse_dates = [column for column in cfg.get('parse_dates')]
        else: 
            parse_dates = None
        # Load csv
        df = pd.read_csv(
            # User defined
            f"{self.config.preprocessor.main_folder}/{cfg['filename']}",
            converters=converters,
            usecols=cfg.get('usecols'),
            names=cfg.get('names'),
            parse_dates=parse_dates,
            # Defaults
            encoding='ISO-8859-1',
            skiprows=[0],
            header=None,
        )

        # Add patients to info dict
        if 'PID' in df.columns:
            self.add_patients_to_info(df)
            # Add admissions to admission dict
            if 'ADMISSION_ID' in df.columns and 'TIMESTAMP' in df.columns:
                self.add_admission_info(df)

        # Apply function
        if cfg.get('function') is not None:
            df = instantiate(cfg['function'])(self, df)
            
        return df

    def save(self, df: pd.DataFrame, filename: str):
        if self.config.preprocessor.file_type == 'parquet':
            df.to_parquet(f'{self.config.preprocessor.output_dir}/{filename}.parquet')
        elif self.config.preprocessor.file_type == 'csv':
            df.to_csv(f'{self.config.preprocessor.output_dir}/{filename}.csv', index=False)

