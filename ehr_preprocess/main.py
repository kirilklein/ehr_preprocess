from os.path import dirname, join, realpath

import hydra
from omegaconf import DictConfig

from ehr_preprocess.preprocessors import mimic, computerome, utils

config_name = "mimic3"
base_dir = dirname(dirname(realpath(__file__)))
config_path = join(base_dir, 'configs')

@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def my_app(cfg: DictConfig) -> None:
    preprocessor = hydra.utils.instantiate(cfg.preprocessor, cfg=cfg)
    preprocessor()
    
    

if __name__=='__main__':
    my_app()