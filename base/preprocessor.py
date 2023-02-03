import pandas as pd
from hydra.utils import instantiate


class BasePreprocessor():
    def __init__(self, 
            config, 
        ):
        self.config = config

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

    def save_concept(self, key: str, df: pd.DataFrame):
        df.to_parquet(f'concept.{key}.parquet')

    def format(self, df: pd.DataFrame, dropna=True):
        # Check if all mandatory columns are present
        for key in self.config.preprocessor.mandatory_columns:
            if key not in df.columns:
                raise ValueError(f'Missing mandatory column {key}')

        # Check if all optional columns are present, if not add them
        for column in self.config.preprocessor.optional_columns:
            if column not in df.columns:
                df[column] = self.config.preprocessor.default_value

        # Drop rows with missing mandatory values
        if dropna:
            df = df[df[self.config.preprocessor.mandatory_columns].notna().all(1)]

        # Reorder columns
        order_columns = self.config.preprocessor.mandatory_columns + self.config.preprocessor.optional_columns
        df = df[order_columns]

        return df

    def load_csv(self, info: dict):
        # Instantiate potential converters
        converters = info.get('converters')
        if converters is not None:
            converters = {column: instantiate(func) for column, func in converters.items()}

        # Load csv
        df = pd.read_csv(
            # User defined
            info['filename'],
            converters=converters,
            usecols=info.get('usecols'),
            names=info.get('names'),
            # Defaults
            encoding='ISO-8859-1',
            skiprows=[0],
            header=None,
        )
            
        return df