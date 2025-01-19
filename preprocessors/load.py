import importlib
from os.path import join

import yaml
from azureml.core import Datastore, Workspace


def instantiate(config, kwargs={}):
    module_path, class_name = config._target_.rsplit(".", 1)
    module = importlib.import_module(module_path)
    class_ = getattr(module, class_name)
    #params = {k: v for k, v in config.items() if k != "_target"}
    instance = class_(**kwargs)
    return instance

def load_config(config_file):
    with open(config_file, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    cfg = Config(cfg)
    return cfg

def load_pandas(self, cfg: dict):
    ds = self.get_dataset(cfg)
    df = ds.to_pandas_dataframe()
    return df

def load_dask(self, cfg: dict):
    ds = self.get_dataset(cfg)
    df = ds.to_dask_dataframe()
    return df

def load_chunks(self, cfg: dict, pandas=True):
    """Generate chunks of the dataset and convert to pandas/dask df"""
    ds = self.get_dataset(cfg)
    i = cfg.start_chunk if 'start_chunk' in cfg else 0
    while True:
        self.logger.info(f"chunk {i}")
        chunk = ds.skip(i * cfg.chunksize)
        chunk = chunk.take(cfg.chunksize)
        df = chunk.to_pandas_dataframe() if pandas else chunk.to_dask_dataframe()
        if len(df.index) == 0:
            self.logger.info("empty")
            break
        i += 1
        yield df
    
class Config(dict):
    def __init__(self, dictionary=None):
        super(Config, self).__init__()
        if dictionary:
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    value = Config(value)
                self[key] = value
                setattr(self, key, value)

    def __setattr__(self, key, value):
        super(Config, self).__setattr__(key, value)
        super(Config, self).__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Config, self).__setitem__(key, value)
        super(Config, self).__setattr__(key, value)

    def __delattr__(self, name):
        if name in self:
            del self[name]
        if hasattr(self, name):
            super(Config, self).__delattr__(name)

    def __delitem__(self, name):
        if name in self:
            del self[name]
        if hasattr(self, name):
            super(Config, self).__delattr__(name)


