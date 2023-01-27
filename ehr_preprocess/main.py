from os.path import dirname, join, realpath

import hydra
from omegaconf import DictConfig

from ehr_preprocess.preprocessors import preprocessors

config_name = "config"
base_dir = dirname(dirname(realpath(__file__)))
config_path = join(base_dir, 'configs')

@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def my_app(cfg: DictConfig) -> None:
    print('config:', cfg)
    hydra.utils.instantiate(cfg)
    
if __name__=='__main__':
    my_app()