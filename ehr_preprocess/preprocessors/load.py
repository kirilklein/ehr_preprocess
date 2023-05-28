import importlib
from types import SimpleNamespace
import yaml
from azureml.core import ScriptRunConfig, Environment, Experiment, Dataset, Datastore, Workspace
from azureml.core.runconfig import MpiConfiguration
from os.path import join

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


def get_datastore():
    # get workspace
    ws = Workspace.from_config()
    # get compute target
    target = ws.compute_targets['Kiril-CPU']
    # get curated environment
    curated_env_name = 'AzureML-PyTorch-1.6-GPU'
    env = Environment.get(workspace=ws, name=curated_env_name)

    subscription_id = 'f8c5aac3-29fc-4387-858a-1f61722fb57a'
    resource_group = 'forskerpl-n0ybkr-rg'
    workspace_name = 'forskerpl-n0ybkr-mlw'
    
    workspace = Workspace(subscription_id, resource_group, workspace_name)
    datastore = Datastore.get(workspace, "researcher_data")
    dump_path = join("data-backup", "SP-dumps", "2022-10-27")
    return datastore, dump_path
