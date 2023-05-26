from preprocessors.base import BasePreprocessor

class AzurePrepocessor(BasePreprocessor):
    # load data in dask
    def __init__(self, cfg, test=False) -> None:
        super().__init__(cfg, test)
        

