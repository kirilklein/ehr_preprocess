from os.path import dirname, join, realpath

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

config_name = "synthea"
base_dir = dirname(dirname(realpath(__file__)))
config_path = join(base_dir, 'configs')

@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def my_app(cfg: DictConfig) -> None:
    preprocessor = instantiate(cfg.preprocessor, cfg=cfg)
    preprocessor()
    
    

if __name__=='__main__':
    my_app()