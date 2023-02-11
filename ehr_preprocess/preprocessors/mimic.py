import json
import os
from os.path import dirname, join, realpath, split

import numpy as np
import pandas as pd

from ehr_preprocess.preprocessors import utils

base_dir = dirname(dirname(dirname(realpath(__file__))))


class BasePreprocessor():
    def __init__(self, cfg, test=False) -> None:
        self.test = test
        self.cfg = cfg
        self.raw_data_path = cfg.paths.raw_data_path
        self.prepends = cfg.prepends
        data_folder_name = split(self.raw_data_path)[-1]

        if not cfg.paths.working_data_path is None:
            working_data_path = cfg.paths.working_data_path
        else:
            working_data_path = join(base_dir, 'data')

        self.interim_data_path = join(
            working_data_path, 'interim', data_folder_name)
        if not os.path.exists(self.interim_data_path):
            os.makedirs(self.interim_data_path)
        self.processed_data_path = join(
            working_data_path, 'processed', data_folder_name)
        if not os.path.exists(self.processed_data_path):
            os.makedirs(self.processed_data_path)
        if test:
            self.nrows = 10000
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
        if os.path.exists(join(self.processed_data_path, 'metadata.json')):
            with open(join(self.processed_data_path, 'metadata.json'), 'r') as fp:
                self.metadata_dic = json.load(fp)
        else:
            self.metadata_dic = {}

    def save_metadata(self):
        with open(join(self.processed_data_path, 'metadata.json'), 'w') as fp:
            json.dump(self.metadata_dic, fp)

    def update_metadata(self, concept_name, coding_system, src_files_ls):
        if concept_name!='patients_info':
            file = f'concept.{concept_name}.parquet'
            prepend = self.prepends[concept_name]
        else:
            file = 'patients_info.parquet'
            prepend = None
        concept_dic = {
            'Coding_System': coding_system, 'Prepend': prepend, 'Source': src_files_ls
        }
        if file not in self.metadata_dic:
            self.metadata_dic[file] = concept_dic

    

class MIMIC3Preprocessor(BasePreprocessor):
    """Extracts events from MIMIC-III database and saves them in a single file."""

    def __init__(self, cfg, test=False):
        super(MIMIC3Preprocessor, self).__init__(cfg, test)
        self.metadata_dic ={
            'patients_info':['', ['ADMISSIONS.csv.gz', 'PATIENTS.csv.gz']],
            'transfer':['', ['ADMISSIONS.csv.gz', 'ICUSTAYS.csv.gz']],
            'diag': ['ICD9', ['DIAGNOSES_ICD.csv.gz', 'ADMISSIONS.csv.gz']],
            'pro':['ICD9', ['PROCEDURES_ICD.csv.gz', 'ADMISSIONS.csv.gz']],
            'med':['DrugName', ['PRESCRIPTIONS.csv.gz']],
            'lab':['LOINC', ['LABEVENTS.csv.gz', 'D_LABITEMS.csv.gz']],
            'event':['', ['OUTPUTEVENTS.csv.gz', 'INPUTEVENTS_MV.csv.gz', 'INPUTEVENTS_CV.csv.gz']],
            'weight':['', ['INPUTEVENTS_MV.csv.gz']]
        }
        self.dtypes = {'SUBJECT_ID':'Int32', 'HADM_ID':'Int32', 'ICUSTAY_ID':'Int32',
            'SEQ_NUM':'Int32', 'PATIENTWEIGHT':float}	
        self.prepend = None

    def __call__(self):
        print('Preprocess Mimic3 from: ', self.raw_data_path)
        if self.cfg.extract_patients_info:
            print(":Extract patient info")
            self.extract_patient_info()
        print(":Extract events")
        for concept_name in self.cfg.concepts:
            print(f"::Extract {concept_name}")
            save_path = join(
                self.processed_data_path,
                f'concept.{concept_name}.parquet')
            preprocessor = globals()[f"MIMICPreprocessor_{concept_name}"](self.cfg, self.test)
            if os.path.exists(save_path):
                if utils.query_yes_no(
                        f"File {save_path} already exists. Overwrite?"):
                    preprocessor()
                    self.update_metadata(concept_name, *self.metadata_dic[concept_name])
                else:
                    print(f"Skipping {concept_name}")
            else:
                preprocessor()
                self.update_metadata(concept_name, *self.metadata_dic[concept_name])
        print(":Save metadata")
        self.save_metadata()

    def extract_patient_info(self):
        patients = pd.read_csv(join(self.raw_data_path, 'PATIENTS.csv.gz'), compression='gzip', nrows=self.nrows,
            ).drop(['ROW_ID', 'DOD_HOSP', 'DOD_SSN', 'EXPIRE_FLAG'], axis=1)
        admissions = self.load_admissions()
        last_admissions = admissions.loc[admissions.groupby('SUBJECT_ID')["ADMITTIME"].idxmax()]
        patient_cols = ['SUBJECT_ID','INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY']
        patients = patients.merge(last_admissions[patient_cols], on=['SUBJECT_ID'], how='left')
        patients = patients.rename(columns={'SUBJECT_ID':'PID', 'DOB':'BIRTHDATE','DOD':'DEATHDATE'})
        patients.to_parquet(join(self.processed_data_path, 'patients_info.parquet'), index=False)
        self.update_metadata('patients_info', '',  ['PATIENTS.csv.gz', 'ADMISSIONS.csv.gz'])
        
    def load_admissions(self):
        admissions = pd.read_csv(join(self.raw_data_path, 'ADMISSIONS.csv.gz'), compression='gzip',
            usecols=['SUBJECT_ID','INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'ADMITTIME'],
            parse_dates=['ADMITTIME'], dtype=self.dtypes)
        return admissions

    def sort_values(self, df):
        return df.sort_values(by=['PID', 'ADMISSION_ID', 'TIMESTAMP', 'CONCEPT'])
    def prepend_concept(self, df):
        df['CONCEPT'] = df['CONCEPT'].map(lambda x: self.prepend + str(x))
        return df
    
class MIMICPreprocessor_transfer(MIMIC3Preprocessor):
    def __init__(self, cfg, test=False):
        super(MIMICPreprocessor_transfer, self).__init__(cfg, test)
        self.concept_name = 'transfer'
    
    def __call__(self):
        df = self.load()
        hospital = self.select_hospital_admissions(df)
        emergency = self.select_emergency_admissions(df)
        icu = self.get_icu_admissions()
        df = pd.concat([hospital, emergency, icu])
        df.rename(columns={'SUBJECT_ID':'PID', 'HADM_ID':'ADMISSION_ID'}, inplace=True)
        df = self.sort_values(df)
        df.to_parquet(join(self.processed_data_path, f'concept.{self.concept_name}.parquet'), index=False)

    def load(self):
        df = pd.read_csv(join(self.raw_data_path, 'ADMISSIONS.csv.gz'), compression='gzip', 
            usecols=['SUBJECT_ID', 'HADM_ID','ADMITTIME', 'DISCHTIME','DEATHTIME',
                'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'ADMISSION_TYPE',
                'EDREGTIME','EDOUTTIME'], 
            parse_dates=['ADMITTIME', 'DISCHTIME', 'EDREGTIME','EDOUTTIME', 'DEATHTIME'],
            dtype=self.dtypes)
        return df

    def select_hospital_admissions(self, df):
        hospital = df.loc[:, ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'ADMISSION_TYPE', 
            'ADMISSION_LOCATION', 'DISCHARGE_LOCATION']]
        hospital.rename(columns={'ADMITTIME': 'TIMESTAMP', 'DISCHTIME': 'TIMESTAMP_END'}, inplace=True)
        hospital['CONCEPT'] = "THOSPITAL"
        return hospital
    def select_emergency_admissions(self, df):
        emergency = df.loc[:, ['SUBJECT_ID', 'HADM_ID', 'EDREGTIME', 'EDOUTTIME']]
        emergency.rename(columns={'EDREGTIME': 'TIMESTAMP', 'EDOUTTIME': 'TIMESTAMP_END'}, inplace=True)
        emergency['CONCEPT'] = "TEMERGENCY"
        # drop rows where both TIMESTAMP and TIMESTAMP_END are null
        emergency = emergency.dropna(subset=['TIMESTAMP', 'TIMESTAMP_END'], how='all')
        return emergency
    def get_icu_admissions(self):
        df_icu = pd.read_csv(join(self.raw_data_path, 'ICUSTAYS.csv.gz'), compression='gzip',
                parse_dates=['INTIME', 'OUTTIME'], nrows=self.nrows, dtype=self.dtypes,
                usecols=['SUBJECT_ID', 'HADM_ID', 'INTIME', 'OUTTIME', 'ICUSTAY_ID'])
        df_icu.rename(columns={'INTIME': 'TIMESTAMP', 'OUTTIME': 'TIMESTAMP_END'}, inplace=True)
        df_icu['CONCEPT'] = "TICU"
        return df_icu

class MIMICPreprocessor_weight(MIMIC3Preprocessor):
    def __init__(self, cfg, test=False):
        super(MIMICPreprocessor_weight, self).__init__(cfg, test)
        self.concept_name = 'weight'
        self.prepend = self.cfg.prepends[self.concept_name]

    def __call__(self):
        weights = self.get_weights()
        weights['CONCEPT'] = 'WEIGHT'
        weights = self.prepend_concept(weights)
        weights.rename(columns={'SUBJECT_ID':'PID', 'HADM_ID':'ADMISSION_ID', 
            'PATIENTWEIGHT':'VALUE', 'STARTTIME':'TIMESTAMP'}, inplace=True)
        weights['VALUE_UNIT'] = 'kg'
        weights.to_parquet(join(self.processed_data_path, f'concept.{self.concept_name}.parquet'), index=False)

    def get_weights(self):
        weights = pd.read_csv(join(self.raw_data_path, f'INPUTEVENTS_MV.csv.gz'), 
            usecols=['SUBJECT_ID', 'HADM_ID', 'STARTTIME', 'PATIENTWEIGHT', 'ICUSTAY_ID'],
            nrows=self.nrows, dtype=self.dtypes, parse_dates=['STARTTIME'])
        return weights

class MIMICPreprocessor_event(MIMIC3Preprocessor):
    def __init__(self, cfg, test=False):
        super(MIMICPreprocessor_event, self).__init__(cfg, test)
        self.concept_name = 'event'
        self.prepend = self.cfg.prepends[self.concept_name]
        self.items_dic = self.get_items_dic()

    def __call__(self):
        events = self.get_chartevents()
        events = pd.merge(events, self.items_dic, on='ITEMID', how='left').drop('ITEMID', axis=1)
        events = events.rename(columns=
            {'SUBJECT_ID':'PID','HADM_ID':'ADMISSION_ID', 'CHARTTIME':'TIMESTAMP', 'VALUEUOM':'VALUE_UNIT', 'LABEL':'CONCEPT'})
        events = self.prepend_concept(events)
        events = self.sort_values(events)
        events.to_parquet(join(self.processed_data_path, f'concept.chart_{self.concept_name}.parquet'), index=False)
        events = self.get_outputevents()
        events = pd.concat([self.get_inputevents_mv(), self.get_inputevents_cv(), self.get_procedureevents_mv(), events])
        events = pd.merge(events, self.items_dic, on='ITEMID', how='left').drop('ITEMID', axis=1)
        events = events.rename(columns=
            {'SUBJECT_ID':'PID','HADM_ID':'ADMISSION_ID', 'CHARTTIME':'TIMESTAMP', 'VALUEUOM':'VALUE_UNIT', 'LABEL':'CONCEPT'})
        events = self.sort_values(events)
        events = self.prepend_concept(events)
        events.to_parquet(join(self.processed_data_path, f'concept.{self.concept_name}.parquet'), index=False)

    def map_itemid_to_label(self, events):
        events['CONCEPT'] = events.ITEMID.map(self.items_dic)
        return events

    def get_chartevents(self):
        events = pd.read_csv(join(self.raw_data_path, 'OUTPUTEVENTS.csv.gz'), 
            usecols=['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM', 'ICUSTAY_ID'],
            nrows=self.nrows, dtype=self.dtypes, parse_dates=['CHARTTIME'])
        return events

    def get_outputevents(self):
        out_events = pd.read_csv(join(self.raw_data_path, 'OUTPUTEVENTS.csv.gz'), 
            usecols=['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM', 'ICUSTAY_ID'],
            nrows=self.nrows, dtype=self.dtypes, parse_dates=['CHARTTIME'])
        out_events = out_events.rename(
            columns={'CHARTTIME': 'TIMESTAMP', 'VALUEOM': 'VALUEUOM'})
        return out_events

    def get_inputevents_mv(self):
        events = pd.read_csv(join(self.raw_data_path, f'INPUTEVENTS_MV.csv.gz'), 
            usecols=['SUBJECT_ID', 'HADM_ID', 'STARTTIME', 'ENDTIME','ITEMID', 'AMOUNT', 'AMOUNTUOM', 'ICUSTAY_ID'],
            nrows=self.nrows, dtype=self.dtypes, parse_dates=['STARTTIME', 'ENDTIME'])
        events = events.rename(columns={
            'STARTTIME': 'TIMESTAMP', 'ENDTIME': 'TIMESTAMP_END', 'AMOUNT': 'VALUE', 'AMOUNTUOM': 'VALUEUOM'})
        return events

    def get_inputevents_cv(self):
        events = pd.read_csv(join(self.raw_data_path, f'INPUTEVENTS_CV.csv.gz'),
            usecols=['SUBJECT_ID', 'HADM_ID', 'CHARTTIME','ITEMID', 'AMOUNT', 'AMOUNTUOM', 'ICUSTAY_ID'],
             nrows=self.nrows, dtype=self.dtypes, parse_dates=['CHARTTIME'])
        events = events.rename(columns={
            'CHARTTIME': 'TIMESTAMP', 'AMOUNT': 'VALUE', 'AMOUNTUOM': 'VALUEUOM'})

    def get_procedureevents_mv(self):
        events = pd.read_csv(join(self.raw_data_path, f'PROCEDUREEVENTS_MV.csv.gz'),
            usecols=['SUBJECT_ID', 'HADM_ID', 'STARTTIME', 'ENDTIME', 'ITEMID', 'ICUSTAY_ID',
                'VALUE', 'VALUEUOM'],
            nrows=self.nrows, dtype=self.dtypes, parse_dates=['STARTTIME', 'ENDTIME'])
        events = events.rename(columns={
            'STARTTIME': 'TIMESTAMP', 'ENDTIME': 'TIMESTAMP_END'})
        return events

    def get_items_dic(self):
        """Get dictionary that maps from ITEMID to LABLES"""
        items_dic = pd.read_csv(join(self.raw_data_path, 'D_ITEMS.csv.gz'),
                usecols=['ITEMID', 'LABEL'], dtype=self.dtypes)
        return items_dic

class MIMICPreprocessor_med(MIMIC3Preprocessor):
    def __init__(self, cfg, test=False):
        super(MIMICPreprocessor_med, self).__init__(cfg, test)
        self.concept_name = 'med'
        self.prepend = self.cfg.prepends[self.concept_name]
    
    def __call__(self):
        df = self.load()
        df = self.rename(df)
        # df = self.handle_range_values(df) we will incorporate it in later preprocessing
        df = self.prepend_concept(df)
        df = self.sort_values(df)
        df.to_parquet(
            join(self.processed_data_path, f'concept.{self.concept_name}.parquet'), index=False)

    def load(self):
        dose_val_rx_converter = lambda x: x.replace(',','.') if ',' in x else x
        df = pd.read_csv(join(self.raw_data_path, 'PRESCRIPTIONS.csv.gz'), compression='gzip', 
                usecols=['SUBJECT_ID', 'HADM_ID','STARTDATE','DOSE_VAL_RX','DOSE_UNIT_RX','DRUG', 'ICUSTAY_ID'],
                parse_dates=['STARTDATE'], converters={'DOSE_VAL_RX': dose_val_rx_converter},
                dtype=self.dtypes, nrows=self.nrows
            ).dropna(subset=['DRUG'])
        return df

    def rename(self, df):
        return df.rename(columns={'SUBJECT_ID': 'PID', 'STARTDATE': 'TIMESTAMP', 'ENDDATE':'TIMESTAMP_END',
            'DRUG': 'CONCEPT', 'DOSE_VAL_RX': 'VALUE', 'DOSE_UNIT_RX': 'VALUE_UNIT', 'HADM_ID': 'ADMISSION_ID'})

    def handle_range_values(self, df):
        """VALUE is often given as range e.g. 1-6, in this case compute mean."""
        df = self.perform_operation_on_column(df, 'VALUE', df.VALUE.str.contains('-'), 
            lambda x: np.average(np.array(x.split('-'), dtype=float)))
        df.VALUE = df.VALUE.astype(float)
        return df

    def perform_operation_on_column(self, df, column, mask, operation):
        """Perform operation on column using loc, with mask and return new column name"""
        df.loc[mask, column] = df.loc[mask, column].map(operation)
        return df
        

class MIMICPreprocessor_pro(MIMIC3Preprocessor):
    def __init__(self, cfg, test=False):
        super(MIMICPreprocessor_pro, self).__init__(cfg, test)
        self.concept_name = 'pro'
        self.prepend = self.cfg.prepends[self.concept_name]

    def __call__(self):
        df = self.load()
        adm_dic = self.load_admission_dic()
        df['TIMESTAMP'] = df['HADM_ID'].map(adm_dic)
        df.rename(
            columns={
                'SUBJECT_ID': 'PID',
                'HADM_ID': 'ADMISSION_ID',
                'ICD9_CODE': 'CONCEPT'},
            inplace=True)
        df = self.prepend_concept(df)
        df = df.rename(columns={'SEQ_NUM':'VALUE'})
        df['VALUE_UNIT'] = 'SEQ_NUM'
        df = self.sort_values(df)
        df.to_parquet(
            join(
                self.processed_data_path,
                f'concept.{self.concept_name}.parquet'),
            index=False)

    def load(self):
        concept_dic = {'pro':'PROCEDURES', 'diag':'DIAGNOSES'}
        df = pd.read_csv(join(self.raw_data_path, f'{concept_dic[self.concept_name]}_ICD.csv.gz'), compression='gzip', 
                nrows=self.nrows, dtype=self.dtypes).drop(columns=['ROW_ID'])
        return df

    def load_admission_dic(self):
        dfa = pd.read_csv(join(self.raw_data_path, 'ADMISSIONS.csv.gz'), compression='gzip', 
                          parse_dates=['ADMITTIME'],
                          usecols=['HADM_ID', 'ADMITTIME'])
        adm_dic = dfa.set_index('HADM_ID').to_dict()['ADMITTIME']
        return adm_dic

class MIMICPreprocessor_diag(MIMICPreprocessor_pro):
    def __init__(self, cfg, test=False):
        super(MIMICPreprocessor_diag, self).__init__(cfg, test)
        self.concept_name = 'diag'
        self.prepend = self.cfg.prepends[self.concept_name]


class MIMICPreprocessor_lab(MIMIC3Preprocessor):

    def __init__(self, cfg, test=False):
        super(MIMICPreprocessor_lab, self).__init__(cfg, test)
        self.concept_name = 'lab'
        self.prepend = self.cfg.prepends[self.concept_name]
    def __call__(self):
        df = self.load()
        df = self.preprocess(df)
        df = df.rename(
            columns={
                'SUBJECT_ID': 'PID',
                'CHARTTIME': 'TIMESTAMP',
                'VALUEUOM': 'VALUE_UNIT',
                'HADM_ID': 'ADMISSION_ID'})
        df = self.sort_values(df)
        df = self.prepend_concept(df)
        df.to_parquet(
            join(
                self.processed_data_path,
                f'concept.{self.concept_name}.parquet'),
            index=False)

    def load(self):
        df = pd.read_csv(join(self.raw_data_path, 'LABEVENTS.csv.gz'),
                         compression='gzip', nrows=self.nrows, parse_dates=['CHARTTIME'],
                         dtype=self.dtypes)
        df = df.drop(columns=['ROW_ID','FLAG', 'VALUENUM'])
        return df

    def preprocess(self, df):
        concept_map = self.get_concept_map()
        df['CONCEPT'] = df.ITEMID.map(concept_map)
        df.drop(columns=['ITEMID', ], inplace=True)
        df.CONCEPT = df.CONCEPT.map(lambda x: self.prepend + x)
        return df

    def load_dic(self):
        return pd.read_csv(
            join(self.raw_data_path, 'D_LABITEMS.csv.gz'), compression='gzip')

    def get_concept_map(self):
        """Map ITEMID to LOINC_CODE or LABEL if LOINC_CODE is missing"""
        df_dic = self.load_dic()
        df_dic = self.preprocess_dic(df_dic)
        item_code_dic = pd.Series(
            df_dic.LOINC_CODE.values,
            index=df_dic.ITEMID).to_dict()
        item_name_dic = pd.Series(df_dic[df_dic.LOINC_CODE.isna(
        )].LABEL.values, index=df_dic[df_dic.LOINC_CODE.isna()].ITEMID).to_dict()
        # combine dicts
        return {**item_code_dic, **item_name_dic}

    def preprocess_dic(self, df_dic):
        # add more specification to SPECIMEN using the fluid column
        df_dic.loc[df_dic['LABEL'].str.contains('SPECIMEN'), 'LABEL'] \
            = df_dic.loc[df_dic['LABEL'].str.contains('SPECIMEN'), 'FLUID']\
            + ' ' + \
            df_dic.loc[df_dic['LABEL'].str.contains('SPECIMEN'), 'CATEGORY']
        return df_dic

# Below are unused functions which might be needed for further preprocessing
    def process_values(self, df):
        df_cont, df_cat = self.separate_continuous_categorical(df)
        df_cont = self.process_continuous_values(df_cont)
        df_cat = self.process_categorical_values(df_cat)
        df = pd.concat([df_cont, df_cat])
        # sort by PID and ADMISSION_ID
        return df

    def process_continuous_values(self, df_cont):
        df_cont.drop(columns=['VALUE'], inplace=True)
        df_cont.rename(columns={'VALUENUM': 'VALUE'}, inplace=True)
        df_cont['VALUE_CAT'] = 'NaN'
        df_cont = df_cont.loc[df_cont['VALUE'] >= 0].copy()
        return df_cont

    def process_categorical_values(self, df_cat):
        df_cat['VALUE_CAT'] = df_cat['VALUE']
        df_cat['VALUE'] = df_cat.groupby('CONCEPT')['VALUE'].transform(
            lambda x: x.astype('category').cat.codes)
        df_cat.drop(columns=['VALUENUM'], inplace=True)
        df_cat['VALUE_UNIT'] = 'categorical'
        return df_cat

    def separate_continuous_categorical(self, df):
        df_cont = df[df['VALUENUM'].notnull()].copy()
        df_cat = df[df['VALUENUM'].isnull()].copy()
        del df
        return df_cont, df_cat

