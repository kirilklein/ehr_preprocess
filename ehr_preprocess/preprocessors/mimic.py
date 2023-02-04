from os.path import dirname, join, split, realpath
import os
import json
from ehr_preprocess.preprocessors import utils
import pandas as pd
from omegaconf import DictConfig, OmegaConf

base_dir = dirname(dirname(dirname(realpath(__file__))))
        

class BasePreprocessor():
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
        self.metadata_ls = []
    def save_metadata(self):
        with open(join(self.processed_data_path, 'metadata.json'), 'w') as fp:
            json.dump(self.metadata_ls, fp)

class MIMIC3Preprocessor(BasePreprocessor):
    """Extracts events from MIMIC-III database and saves them in a single file."""
    def __init__(self, cfg, test=False):
        super(MIMIC3Preprocessor, self).__init__(cfg, test)

    def __call__(self):
        print('Preprocess Mimic3 from: ', self.raw_data_path)
        print(":Extract patient info")
        self.extract_patient_info()
        print(":Extract events")
        self.extract_events()
        self.save_metadata()

    def extract_events(self):
        self.extract_lab()
        pass

    def extract_patient_info():
        pass
        
    def extract_lab(self):
        print("::Extract lab results")
        MIMICLabPreprocessor(self.cfg, self.test)()
        self.metadata_ls.append({
            'Type':'Lab', 'System':'LOINC', 'File':'lab.parquet', 'Prepend':'L'
        })

    def extract_diagnoses():
        pass
        
    def extract_prescriptions():
        pass

    def extract_procedures():
        pass

class MIMICLabPreprocessor(MIMIC3Preprocessor):

    def __init__(self, cfg, test=False):
        super(MIMICLabPreprocessor, self).__init__(cfg, test)
    
    def __call__(self):
        df = self.load()
        df_dic = self.load_dic()
        df = self.preprocess(df, df_dic)
        df.to_parquet(join(self.processed_data_path, 'lab.parquet'))

    def load(self):
        df = pd.read_csv(join(self.raw_data_path, 'LABEVENTS.csv.gz'), 
            compression='gzip', nrows=self.nrows, parse_dates=['CHARTTIME'], 
            dtype={'SUBJECT_ID': 'Int32', 'ITEMID': 'Int32', 'VALUE': 'str', 'VALUENUM': 'float32', 'VALUEUOM': 'str', 'HADM_ID': 'Int32'})
        df = df.rename(columns={'SUBJECT_ID': 'PID', 'CHARTTIME': 'TIMESTAMP', 'VALUEUOM': 'VALUE_UNIT', 'HADM_ID': 'ADMISSION_ID'}).drop(columns=['ROW_ID', 'FLAG'])
        return df
    
    def load_dic(self):
        return pd.read_csv(join(self.raw_data_path, 'D_LABITEMS.csv.gz'), compression='gzip')

    def preprocess_dic(self, df_dic):
        # add more specification to SPECIMEN using the fluid column
        df_dic.loc[df_dic['LABEL'].str.contains('SPECIMEN'), 'LABEL'] \
            = df_dic.loc[df_dic['LABEL'].str.contains('SPECIMEN'), 'FLUID']\
                + ' ' + df_dic.loc[df_dic['LABEL'].str.contains('SPECIMEN'), 'CATEGORY']
        return df_dic
    
    def preprocess(self, df, df_dic):
        concept_map = self.get_concept_map(df_dic)
        df['CONCEPT'] = df.ITEMID.map(concept_map)
        df.drop(columns=['ITEMID'], inplace=True)
        df = self.process_values(df)
        pass
        
    def get_concept_map(self, df_dic):
        item_code_dic = pd.Series(df_dic.LOINC_CODE.values, index=df_dic.ITEMID).to_dict()
        item_name_dic = pd.Series(df_dic[df_dic.LOINC_CODE.isna()].LABEL.values, index=df_dic[df_dic.LOINC_CODE.isna()].ITEMID).to_dict()
        # combine dicts
        return {**item_code_dic, **item_name_dic}

    def process_values(self, df):
        df_cont = self.process_continuous_values(df)
        df_cat = self.process_categorical_values(df)
        df = pd.concat([df_cont, df_cat])
        # sort by PID and ADMISSION_ID
        df.sort_values(by=['PID', 'ADMISSION_ID', 'TIMESTAMP'], inplace=True)
        df.CONCEPT = df.CONCEPT.map(lambda x: 'L'+x)
        df.to_parquet(join(self.processed_data_path, 'concept.lab.parquet'))

    def process_continuous_values(df):
        df_cont = df[df['VALUENUM'].notnull()]
        df_cont = df_cont[df_cont['VALUENUM'] >= 0]
        df_cont.drop(columns=['VALUE'], inplace=True)
        df_cont.rename(columns={'VALUENUM': 'VALUE'}, inplace=True)
        df_cont.drop(columns=['VALUE_NUM'], inplace=True)
        df_cont['VALUE_CAT'] = 'NaN'
        return df_cont

    def process_categorical_values(df):
        df_cat = df[df['VALUENUM'].isnull()]
        df_cat['VALUE_CAT'] = df_cat['VALUE']
        df_cat['VALUE'] = df_cat.groupby('ITEMID')['VALUE'].transform(lambda x: x.astype('category').cat.codes)
        df_cat.drop(columns=['VALUE_NUM'], inplace=True)
        df_cat['VALUE_UNIT'] = 'categorical'
        return df_cat
    
if __name__ == '__main__':
    cfg = OmegaConf.load(join(base_dir, "configs", "mimic3.yaml"))
    preprocessor = MIMIC3Preprocessor(cfg, test=True)
    