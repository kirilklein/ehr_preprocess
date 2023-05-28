from os.path import dirname, join, realpath

from preprocessors.load import instantiate, load_config
import yaml

config_name = "azure"
def my_app(config_name):
    base_dir = dirname(dirname(realpath(__file__)))
    config_path = join(base_dir, 'configs')
    cfg = load_config(join(config_path, config_name+'.yaml'))
    preprocessor = instantiate(cfg.preprocessor, {'cfg':cfg})
    preprocessor()
    
    

if __name__=='__main__':
    my_app(config_name)