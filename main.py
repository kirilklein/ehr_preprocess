from os.path import dirname, join, realpath
import shutil
from preprocessors.load import instantiate, load_config
import logging
from azure_run.run import Run
from azure_run import datastore
import pathlib

run = Run
run.name(f"Diagnosis_Medication_Labtest_Procedure")
ds_sp = datastore("researcher_data")

config_name = "azure"
def my_app(config_name):
    base_dir = dirname(realpath(__file__))
    config_path = join(base_dir, 'configs')
    cfg = load_config(join(config_path, config_name+'.yaml'))
    pathlib.Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True) # added line
    shutil.copyfile(join(config_path, config_name+'.yaml'), join(cfg.paths.output_dir, 'config.yaml'))
    logging.basicConfig(filename=join(cfg.paths.output_dir, cfg.run_name+'.log'), level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(__name__)
    print(cfg.paths.dump_path)
    preprocessor = instantiate(cfg.preprocessor, {'cfg':cfg, 'logger':logger, 'datastore':ds_sp, 'dump_path': cfg.paths.dump_path})
    preprocessor()    

if __name__=='__main__':
    my_app(config_name)