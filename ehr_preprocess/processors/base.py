import pandas as pd
from hydra.utils import instantiate


class BasePreprocessor():
    def __init__(self, 
            config, 
        ):
        self.config = config
        self.info = {}

    def __call__(self):
        self.process()

    # All saving is handled internally in the functions below
    def process(self):
        self.concepts()
        self.patients_info()

    def concepts(self):
        raise NotImplementedError

    def patients_info(self):
        raise NotImplementedError

    def set_patient_info(self, df: pd.DataFrame):
        df = df.set_index('PID')
        df.apply(lambda patient:
            self.info.get(patient.name)             # patient.name is the index PID
            .update(
                {k: v for k,v in patient.items()}   # Add every column to patient info
            ), axis=1
        )

    def save(self, df: pd.DataFrame, filename: str):
        if self.config.preprocessor.format == 'parquet':
            df.to_parquet(f'{self.config.preprocessor.output_dir}/{filename}.parquet')
        elif self.config.preprocessor.format == 'csv':
            df.to_csv(f'{self.config.preprocessor.output_dir}/{filename}.csv', index=False)

    def add_patients_to_info(self, df: pd.DataFrame):
        for pat in df['PID'].unique():
            self.info.setdefault(pat, {})

    def info_to_df(self):
        df = pd.DataFrame.from_dict(self.info, orient="index")
        df = df.reset_index().rename(columns={'index': 'PID'})    # Convert the PID index to PID column
        return df

    def format(self, df: pd.DataFrame, dropna=True):
        # Check if all mandatory columns are present
        for key in self.config.preprocessor.columns.mandatory:
            if key not in df.columns:
                raise ValueError(f'Missing mandatory column {key}')

        # Check if all optional columns are present, if not add them
        for column in self.config.preprocessor.columns.optional:
            if column not in df.columns:
                df[column] = self.config.preprocessor.default_value

        # Drop rows with missing mandatory values
        if dropna:
            df = df[df[self.config.preprocessor.columns.mandatory].notna().all(1)]

        # Reorder columns
        order_columns = self.config.preprocessor.columns.mandatory + self.config.preprocessor.columns.optional
        df = df[order_columns]

        return df

    def load_csv(self, info: dict):
        # Instantiate potential converters
        converters = info.get('converters')
        if converters is not None:
            converters = {column: instantiate(func) for column, func in converters.items()}

        # Conversion from ListConfig to List
        if info.get('date_columns') is not None:
            parse_dates = [col for col in info.get('date_columns')]
        else:
            parse_dates = None
            
        # Load csv
        df = pd.read_csv(
            # User defined
            f"{self.config.preprocessor.main_folder}/{info['filename']}",
            converters=converters,
            usecols=info.get('usecols'),
            names=info.get('names'),
            parse_dates=parse_dates,
            # Defaults
            encoding='ISO-8859-1',
            skiprows=[0],
            header=None,
        )

        # Add patients to info dict
        if 'PID' in df.columns:
            self.add_patients_to_info(df)
            
        return df