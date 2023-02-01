import pandas as pd
from hydra.utils import instantiate


class BasePreprocessor():
    def __init__(self, main_folder: str=None, config=None):
        self.main_folder = main_folder
        self.config = config

    def process(self):
        raise NotImplementedError

    def concepts(self):
        raise NotImplementedError

    def patients_info(self):
        raise NotImplementedError

    def load_csv(info: dict):
        # Instantiate potential converters
        converters = info.get('converters', None)
        if converters is not None:
            converters = {column: instantiate(func) for column, func in converters.items()}

        df = pd.read_csv(
            # User defined
            info['filename'],
            converters=converters,
            usecols=info.get('usecols', None),
            names=info.get('names', None),
            # Defaults
            encoding='ISO-8859-1',
            skiprows=[0],
            header=None,
        )
        return df