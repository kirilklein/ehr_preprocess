from os.path import dirname, join, realpath
import shutil
from preprocessors.load import instantiate, load_config
import logging
from azure_run.run import Run
from azure_run import datastore


run = Run
run.name(f"Diagnoses_Medication_Debugged")
ds_sp = datastore("sp_data")

config_name = "azure"
def my_app(config_name):
    # datastore = Datastore.get(ws, 'workspaceblobstore')
    base_dir = dirname(realpath(__file__))
    config_path = join(base_dir, 'configs')
    cfg = load_config(join(config_path, config_name+'.yaml'))
    shutil.copyfile(join(config_path, config_name+'.yaml'), join(cfg.paths.output_dir, 'config.yaml'))
    logging.basicConfig(filename=join(cfg.paths.output_dir, cfg.run_name+'.log'), level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(__name__)
    preprocessor = instantiate(cfg.preprocessor, {'cfg':cfg, 'logger':logger, 'datastore':ds_sp})
    preprocessor()
    
        

if __name__=='__main__':
    my_app(config_name)