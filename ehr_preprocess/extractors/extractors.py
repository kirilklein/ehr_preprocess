from os.path import dirname, join, split, realpath
import os
from ehr_preprocess.preprocessors import utils

base_dir = dirname(dirname(dirname(realpath(__file__))))

class BasePreprocessor:
    def __init__(self, cfg):
        self.cfg = cfg
        if not cfg.data.raw_data_path is None:
            self.raw_data_path = cfg.data.raw_data_path
        else:
            self.raw_data_path = join(base_dir, 'data', 'raw', 'mimic-iii-clinical-database-1.4')
    def forward(self):
        raise NotImplementedError


class MimicIII(BasePreprocessor): 
    def __init__(self, cfg):
        super(MimicIII, self).__init__(cfg)
    
    def forward(self):
        print('Preprocess MimicIII from: ', self.raw_data_path)
        


class MIMIC3Extractor():
    """Extracts events from MIMIC-III database and saves them in a single file."""
    def __init__(self, cfg, test=False):
        self.cfg = cfg
        self.raw_data_dir = cfg.data.raw_data_dir
        data_folder_name = split(self.raw_data_dir)[-1]
        
        if not cfg.data.working_data_dir is None:
            working_data_dir = cfg.data.working_data_dir
        else: 
            working_data_dir = join(base_dir, 'data')
        
        self.interim_data_dir = join(working_data_dir, 'interim', 'mimic-iii-clinical-database-1.4')
        if not os.path.exists(self.interim_data_dir):
            os.makedirs(self.interim_data_dir)
        self.processed_data_dir = join(working_data_dir, 'processed', 'mimic-iii-clinical-database-1.4')
        if not os.path.exists(self.processed_data_dir):
            os.makedirs(self.processed_data_dir)
        if test:
            self.nrows = 1000
        else:
            self.nrows = None
    def forward(self):
        self.extract_patient_info()
        print(":Extract events")
        print("::Extract diagnoses")
        print("::Extract prescriptions")
        print("::Extract lab results")
        print("::Extract vital signs")
        print("::Extract admissions")
        print('Preprocess Mimic3 from: ', self.raw_data_path)

    def extract_events():
        pass

    def extract_patient_info():
        print(":Extract patient info")
        

    def extract_diagnoses():
        pass

    def extract_prescriptions():
        pass

    def extract_procedures():
        pass

    