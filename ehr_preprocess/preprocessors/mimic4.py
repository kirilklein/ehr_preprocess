from preprocessors import base
import pandas as pd
from os.path import join

class MIMIC4Preprocessor(base.BasePreprocessor):
    """Extracts events from MIMIC-III database and saves them in a single file."""

    def __init__(self, cfg, test=True):
        super(MIMIC4Preprocessor, self).__init__(cfg)
        self.test = test
        self.data_dir = cfg.paths.main_folder
        self.output_dir = cfg.paths.output_dir
        self.set_rename_dic()

    def __call__(self):
        self.patients_info()
        self.concepts()

    def patients_info(self):
        patients = self.load_basic_info()
        patients = self.add_admission_info(patients)
        
        patients = patients.rename(columns=self.rename_dic)
        patients.columns = patients.columns.str.upper()
        self.save(patients, 'patients_info.csv')
    
    def concepts(self):
        for cfg in self.config.concepts:
            df = self.load_csv(cfg)
            df = df.rename(columns=self.rename_dic)
            df.columns = df.columns.str.upper()
            self.save(df, f'concepts.{cfg.name}.csv')

    def load_csv(self, cfg):
        return pd.read_csv(join(self.data_dir, cfg.filename),
                           usecols=cfg.load_columns, 
                           nrows=1000 if self.test else None)
    
    def load_basic_info(self):
        patients = self.load_csv(self.config.patients_info.basic_info)
        patients = self.calculate_birthdates(patients)
        return patients

    def add_admission_info(self, patients):
        cfg = self.config.patients_info.admission_info
        admissions = self.load_csv(cfg)
        for col in cfg.load_columns:
            if col=='subject_id':
                continue
            add_info = admissions.groupby('subject_id')[col].agg(lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else None).reset_index()
            patients = patients.merge(add_info, on='subject_id', how='left')
        return patients

    @staticmethod
    def calculate_birthdates(patients):
        patients['DATE_OF_BIRTH'] = patients['anchor_year'] - patients['anchor_age']
        patients = patients.drop(columns=['anchor_year', 'anchor_age'])
        patients['DATE_OF_BIRTH'] = pd.to_datetime(patients['DATE_OF_BIRTH'], format='%Y')
        return patients
    
    def set_rename_dic(self):
        self.rename_dic = {
            'subject_id': 'PID',
            'dod':'DATE_OF_DEATH',
        }
   
