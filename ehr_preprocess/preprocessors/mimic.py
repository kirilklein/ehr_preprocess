from os.path import dirname, join, split, realpath
import os
from ehr_preprocess.processors import utils
import pandas as pd
from omegaconf import DictConfig
base_dir = dirname(dirname(dirname(realpath(__file__))))
        

class BaseProcessor():
    def __init__(self, cfg, test=False) -> None:
        self.cfg = cfg
        self.raw_data_path = cfg.paths.raw_data_path
        data_folder_name = split(self.raw_data_path)[-1]
        
        if not cfg.paths.working_data_path is None:
            working_data_path = cfg.paths.working_data_path
        else: 
            working_data_path = join(base_dir, 'data')
        
        self.interim_data_path = join(working_data_path, 'interim', data_folder_name)
        if not os.path.exists(self.interim_data_path):
            os.makedirs(self.interim_data_path)
        self.processed_data_path = join(working_data_path, 'processed', data_folder_name)
        if not os.path.exists(self.processed_data_path):
            os.makedirs(self.processed_data_path)
        else:
            if len(os.listdir(self.processed_data_path)) > 0:
                utils.query_yes_no(f"Processed data folder {self.processed_data_path} already exists and contains files. Files may be overwritten. Continue?")
        if test:
            self.nrows = 1000
        else:
            self.nrows = None
        self.rename_dic = {
            'SUBJECT_ID': 'pid', 
            'ITEMID': 'itemid', 
            'CHARTTIME': 'timestamp', 
            'VALUE': 'value', 
            'VALUENUM': 'valuenum', 
            'VALUEUOM': 'unit'
        }
        self.metadata_dic = {'Type':[], 'System':[]}

class MIMIC3Processor(BaseProcessor):
    """Extracts events from MIMIC-III database and saves them in a single file."""
    def __init__(self, cfg, test=False):
        super(MIMIC3Processor, self).__init__(cfg, test)

    def forward(self):
        self.extract_patient_info()
        print(":Extract events")
        print("::Extract diagnoses")
        print("::Extract prescriptions")
        print("::Extract lab results")
        print("::Extract vital signs")
        print("::Extract admissions")
        print('Preprocess Mimic3 from: ', self.raw_data_path)
        self.save_metadata()
    def extract_events():
        pass

    def extract_patient_info():
        print(":Extract patient info")
        
    def extract_lab(self):
        df = pd.read_csv(join(self.raw_data_path, 'LABEVENTS.csv'), 
            nrows=self.nrows, usecols=['SUBJECT_ID', 'ITEMID', 'CHARTTIME', 'VALUE', 'VALUENUM','VALUEUOM'],
            parse_dates=['CHARTTIME'], dtype={'SUBJECT_ID': 'Int32', 'ITEMID': 'Int32', 'VALUE': 'str', 'VALUENUM': 'float32', 'VALUEUOM': 'str'},
            compression='gzip')
        df = df.rename(columns=self.rename_dic)
        self.metadata_dic['Type'].append('Lab')
        self.metadata_dic['System'].append('LOINC')

    def extract_diagnoses():
        pass
    
    def save_metadata(self):
        pd.DataFrame(self.metadata_dic).to_parquet(join(self.processed_data_path, 'metadata.parquet'))
        
    def extract_prescriptions():
        pass

    def extract_procedures():
        pass

    
if __name__ == '__main__':
    cfg = {'data': {'raw_data_dir': 'C:\\Users\\user\\Documents\\GitHub\\ehr_preprocess\\data\\raw\\mimic-iii-clinical-database-1.4'}}
    omegacfg = DictConfig(cfg)
    processor = MIMIC3Processor(omegacfg)
    