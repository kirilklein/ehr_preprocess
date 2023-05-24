import os
from os.path import join

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tqdm import tqdm

from ehr_preprocess.preprocessors import base, utils

class MIMIC3Preprocessor(base.BaseMIMICPreprocessor):
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
            'chartevent':['', ['OUTPUTEVENTS.csv.gz', 'INPUTEVENTS_MV.csv.gz', 'INPUTEVENTS_CV.csv.gz', 'CHARTEVENTS.csv.gz', 'PROCEDUREEVENTS_MV.csv.gz']],
            'microbio':['', ['MICROBIOLOGYEVENTS.csv.gz']],
            'weight':['', ['INPUTEVENTS_MV.csv.gz']]
        }
        self.dtypes = {'SUBJECT_ID':'Int32', 'HADM_ID':'Int32', 'ICUSTAY_ID':'Int32',
            'SEQ_NUM':'Int32', 'PATIENTWEIGHT':float,}	
        self.rename_dic = {
            'SUBJECT_ID': 'PID',
            'HADM_ID':'ADMISSION_ID',
            'STARTTIME':'TIMESTAMP',
            'ENDTIME':'TIMESTAMP_END',
            'CHARTTIME': 'TIMESTAMP',
            'VALUEUOM': 'VALUE_UNIT',
            'PATIENTWEIGHT':'VALUE',
            'LABEL':'CONCEPT',
        }
    @utils.timing_function
    def __call__(self):
        print('Preprocess Mimic3 from: ', self.raw_data_path)
        save_path = join(self.formatted_data_path, 'patients_info.parquet')
        if self.cfg.extract_patients_info:
            print(":Extract patient info")
            if os.path.exists(save_path):
                if utils.query_yes_no(f"File {save_path} already exists. Overwrite?"):
                    self.extract_patient_info(save_path)
            else:
                self.extract_patient_info(save_path)

        print(":Extract events")
        for concept_name in self.cfg.concepts:
            print(f" ::Extract {concept_name}")
            save_path = join(
                self.formatted_data_path,
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
    
    def extract_patient_info(self, save_path):
        print("  ::: Load PATIENTS")
        patients = pd.read_csv(join(self.raw_data_path, 'PATIENTS.csv.gz'), compression='gzip', nrows=self.nrows,
            parse_dates=['DOB', 'DOD'], infer_datetime_format=True,
            ).drop(['ROW_ID', 'DOD_HOSP', 'DOD_SSN', 'EXPIRE_FLAG'], axis=1)
        admissions = self.load_admissions()
        last_admissions = admissions.loc[admissions.groupby('SUBJECT_ID')["ADMITTIME"].idxmax()]
        patient_cols = ['SUBJECT_ID','INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY']
        patients = patients.merge(last_admissions[patient_cols], on=['SUBJECT_ID'], how='left')
        patients = patients.rename(columns={'SUBJECT_ID':'PID', 'DOB':'BIRTHDATE','DOD':'DEATHDATE'})
        patients['BIRTHDATE'] = pd.to_datetime(patients['BIRTHDATE'], format='%Y-%m-%d %H:%M:%S')
        patients['DEATHDATE'] = pd.to_datetime(patients['DEATHDATE'], format='%Y-%m-%d %H:%M:%S')
        patients = patients.reset_index(drop=True)
        pq.write_table(pa.Table.from_pandas(patients), save_path)
        self.update_metadata('patients_info', '',  ['PATIENTS.csv.gz', 'ADMISSIONS.csv.gz'])
        
    def load_admissions(self):
        print("  ::: Load admissions")
        admissions = pd.read_csv(join(self.raw_data_path, 'ADMISSIONS.csv.gz'), compression='gzip',
            usecols=['SUBJECT_ID','INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'ADMITTIME'],
            parse_dates=['ADMITTIME'], infer_datetime_format=True, dtype=self.dtypes)
        return admissions
    @staticmethod
    def sort_values(df):
        return df.sort_values(by=['PID', 'ADMISSION_ID', 'TIMESTAMP', 'CONCEPT'])

    def prepend_concept(self, df):
        df['CONCEPT'] = df['CONCEPT'].map(lambda x: self.prepend + str(x))
        return df

    def write_concept_to_parquet(self, df, suffix=''):
        pq.write_table(pa.Table.from_pandas(df), join(self.formatted_data_path, f'concept.{self.concept_name}{suffix}.parquet'))


class MIMICEventPreprocessor(MIMIC3Preprocessor):
    def __init__(self, cfg, concept_name, test=False,):
        super(MIMICEventPreprocessor, self).__init__(cfg, test)
        self.prepend = self.cfg.prepends[concept_name]
    @utils.timing_function
    def __call__(self, df, suffix=''):
        df = self.sort_values(df)
        df = self.prepend_concept(df)
        df = df.reset_index(drop=True)
        self.write_concept_to_parquet(df, suffix)


class MIMICPreprocessor_transfer(MIMICEventPreprocessor):
    def __init__(self, cfg, test=False):
        self.concept_name = 'transfer'
        super(MIMICPreprocessor_transfer, self).__init__(cfg, self.concept_name, test)
    @utils.timing_function
    def __call__(self):
        transfers = self.load()
        hospital = self.select_hospital_admissions(transfers)
        emergency = self.select_emergency_admissions(transfers)
        icu = self.get_icu_admissions()
        transfers = pd.concat([hospital, emergency, icu])
        transfers.rename(columns=self.rename_dic, inplace=True)
        super().__call__(transfers)

    def load(self):
        print("  ::: load ADMISSIONS")
        df = pd.read_csv(join(self.raw_data_path, 'ADMISSIONS.csv.gz'), compression='gzip', 
            usecols=['SUBJECT_ID', 'HADM_ID','ADMITTIME', 'DISCHTIME','DEATHTIME',
                'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'ADMISSION_TYPE',
                'EDREGTIME','EDOUTTIME'], 
            parse_dates=['ADMITTIME', 'DISCHTIME', 'EDREGTIME','EDOUTTIME', 'DEATHTIME'],
            infer_datetime_format=True,
            dtype=self.dtypes)
        return df

    def select_hospital_admissions(self, df):
        hospital = df.loc[:, ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'ADMISSION_TYPE', 
            'ADMISSION_LOCATION', 'DISCHARGE_LOCATION']]
        hospital.rename(columns={'ADMITTIME': 'TIMESTAMP', 'DISCHTIME': 'TIMESTAMP_END'}, inplace=True)
        hospital['CONCEPT'] = "HOSPITAL"
        return hospital
        
    def select_emergency_admissions(self, df):
        emergency = df.loc[:, ['SUBJECT_ID', 'HADM_ID', 'EDREGTIME', 'EDOUTTIME']]
        emergency.rename(columns={'EDREGTIME': 'TIMESTAMP', 'EDOUTTIME': 'TIMESTAMP_END'}, inplace=True)
        emergency['CONCEPT'] = "EMERGENCY"
        # drop rows where both TIMESTAMP and TIMESTAMP_END are null
        emergency = emergency.dropna(subset=['TIMESTAMP', 'TIMESTAMP_END'], how='all')
        return emergency
    def get_icu_admissions(self):
        print("  ::: load ICUSTAYS")
        df_icu = pd.read_csv(join(self.raw_data_path, 'ICUSTAYS.csv.gz'), compression='gzip',
                parse_dates=['INTIME', 'OUTTIME'], nrows=self.nrows, dtype=self.dtypes,
                usecols=['SUBJECT_ID', 'HADM_ID', 'INTIME', 'OUTTIME', 'ICUSTAY_ID'])
        df_icu.rename(columns={'INTIME': 'TIMESTAMP', 'OUTTIME': 'TIMESTAMP_END'}, inplace=True)
        df_icu['CONCEPT'] = "ICU"
        return df_icu

class MIMICPreprocessor_weight(MIMICEventPreprocessor):
    def __init__(self, cfg, test=False):
        self.concept_name = 'weight'
        super(MIMICPreprocessor_weight, self).__init__(cfg, self.concept_name, test)
    @utils.timing_function
    def __call__(self):
        weights = self.get_weights()
        weights['CONCEPT'] = 'WEIGHT'
        weights.rename(columns=self.rename_dic, inplace=True)
        weights['VALUE_UNIT'] = 'kg'
        super().__call__(weights)    

    def get_weights(self):
        print("  ::: load INPUTEVENTS_MV")
        weights = pd.read_csv(join(self.raw_data_path, f'INPUTEVENTS_MV.csv.gz'), 
            usecols=['SUBJECT_ID', 'HADM_ID', 'STARTTIME',  'PATIENTWEIGHT', 'ICUSTAY_ID'],
            nrows=self.nrows, dtype=self.dtypes, parse_dates=['STARTTIME'], infer_datetime_format=True,)
        # could be improved by sharing inputevents_mv with chartevents
        return weights


class MIMICPreprocessor_chartevent(MIMICEventPreprocessor):
    def __init__(self, cfg, test=False):
        self.concept_name = 'chartevent'
        super(MIMICPreprocessor_chartevent, self).__init__(cfg, self.concept_name, test)
        self.items_dic = self.get_items_dic()
        self.usecols_dic = {'output':['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM', 'ICUSTAY_ID'],
                    'input_cv':['SUBJECT_ID', 'HADM_ID', 'CHARTTIME','ITEMID', 'AMOUNT', 'AMOUNTUOM', 'ICUSTAY_ID'],
                    'input_mv':['SUBJECT_ID', 'HADM_ID', 'STARTTIME', 'ENDTIME','ITEMID', 'AMOUNT', 'AMOUNTUOM', 'ICUSTAY_ID'],
                    'procedure':['SUBJECT_ID', 'HADM_ID', 'STARTTIME', 'ENDTIME', 'ITEMID', 'VALUE', 'VALUEUOM', 'ICUSTAY_ID'],}
        self.usecols_dic['chart'] = self.usecols_dic['output'] 
        self.event_rename_dic = {
            'CHARTTIME': 'TIMESTAMP', 'STARTTIME': 'TIMESTAMP', 'ENDTIME': 'TIMESTAMP_END', 
            'AMOUNT': 'VALUE', 'AMOUNTUOM': 'VALUE_UNIT', 'VALUEUOM': 'VALUE_UNIT'}
        self.load_kwargs = {'infer_datetime_format': True, 'nrows':self.nrows, 'dtype':self.dtypes}
    @utils.timing_function
    def __call__(self):
        self.forward_main_chartevents()
        self.forward_other_chartevents()
    
    def forward_main_chartevents(self):
        """forward the large chartevents table, iteration through the table is necessary"""
        all_chartevents = []
        chartevents_iter = self.get_chartevents()
        for chartevent in tqdm(chartevents_iter):
            chartevent = self.forward_table(chartevent)
            chartevent = self.prepend_concept(chartevent)
            chartevent['VALUE'] = chartevent['VALUE'].astype(str)
            all_chartevents.append(chartevent)
        all_chartevents = pd.concat(all_chartevents)
        all_chartevents = self.sort_values(all_chartevents)
        all_chartevents = all_chartevents.reset_index(drop=True)
        # turn VALUE column to str, for saving
        self.write_concept_to_parquet(all_chartevents, '_main')

    def forward_other_chartevents(self):
        """iterate through other events (procedure, output, input_cv, input_mv)"""
        all_events = []
        for get_func in [self.get_procedure_events, self.get_outputevents, 
                        self.get_input_cv_events, self.get_input_mv_events]:
            all_events.append(self.forward_table(get_func()))
        all_events = pd.concat(all_events)
        super().__call__(all_events)
        del all_events
        

    def forward_table(self, events):
        events = events.rename(columns=self.event_rename_dic)
        events = pd.merge(events, self.items_dic, on='ITEMID', how='left').drop('ITEMID', axis=1)
        events = events.rename(columns=self.rename_dic)
        return events

    def map_itemid_to_label(self, events):
        events['CONCEPT'] = events.ITEMID.map(self.items_dic)
        return events
    @utils.timing_function
    def get_chartevents(self):
        print('  ::: load CHARTEVENTS')
        return pd.read_csv(join(self.raw_data_path, 'CHARTEVENTS.csv.gz'), chunksize=100 if self.test else int(10e6),
            usecols=self.usecols_dic['chart'], parse_dates=['CHARTTIME'], **self.load_kwargs)
    @utils.timing_function
    def get_outputevents(self):
        print('  ::: load OUTPUTEVENTS')
        return pd.read_csv(join(self.raw_data_path, 'OUTPUTEVENTS.csv.gz'), 
            usecols=self.usecols_dic['output'], parse_dates=['CHARTTIME'], **self.load_kwargs)
    @utils.timing_function
    def get_input_mv_events(self):
        print('  ::: load INPUTEVENTS_MV')
        return pd.read_csv(join(self.raw_data_path, f'INPUTEVENTS_MV.csv.gz'), 
            usecols=self.usecols_dic['input_mv'], parse_dates=['STARTTIME', 'ENDTIME'],  **self.load_kwargs)
    @utils.timing_function
    def get_procedure_events(self):
        print('  ::: load PROCEDUREEVENTS_MV')
        return pd.read_csv(join(self.raw_data_path, f'PROCEDUREEVENTS_MV.csv.gz'),
            usecols=self.usecols_dic['procedure'], parse_dates=['STARTTIME', 'ENDTIME'], **self.load_kwargs)
    @utils.timing_function
    def get_input_cv_events(self):
        print('  ::: load INPUTEVENTS_CV')
        return pd.read_csv(join(self.raw_data_path, f'INPUTEVENTS_CV.csv.gz'),
            usecols=self.usecols_dic['input_cv'], parse_dates=['CHARTTIME'], **self.load_kwargs)
    @utils.timing_function
    def get_items_dic(self):
        """Get dictionary that maps from ITEMID to LABLES"""
        print('  ::: load D_ITEMS')
        items_dic = pd.read_csv(join(self.raw_data_path, 'D_ITEMS.csv.gz'),
                usecols=['ITEMID', 'LABEL'], dtype=self.dtypes)
        return items_dic


class MIMICPreprocessor_microbio(MIMICEventPreprocessor):
    def __init__(self, cfg, test=False):
        self.concept_name = 'microbio'
        super(MIMICPreprocessor_microbio, self).__init__(cfg, self.concept_name, test)
    @utils.timing_function
    def __call__(self):
        mbio = self.load()
        # replace nan with empty string
        mbio.ORG_NAME.fillna('', inplace=True)
        mbio.SPEC_TYPE_DESC.fillna('', inplace=True)
        mbio = self.get_concept(mbio)
        mbio = mbio.rename(columns={'CHARTTIME':'TIMESTAMP', 'INTERPRETATION':'VALUE'})
        mbio.rename(columns=self.rename_dic, inplace=True)
        super().__call__(mbio)

    def load(self):
        print("  ::: load MICROBIOLOGYEVENTS")
        mbio = pd.read_csv(join(self.raw_data_path,'MICROBIOLOGYEVENTS.csv.gz'), compression='gzip', 
                nrows=self.nrows, dtype=self.dtypes, 
                usecols=['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'SPEC_TYPE_DESC', 'ORG_NAME', 'INTERPRETATION'])
        return mbio
    @staticmethod
    def get_concept(mbio):
        """We use the two most informative columns to get the concept"""
        mbio['CONCEPT'] = mbio['SPEC_TYPE_DESC'] + '_' + mbio['ORG_NAME']
        mbio.drop(['SPEC_TYPE_DESC', 'ORG_NAME'], axis=1, inplace=True)
        return mbio


class MIMICPreprocessor_med(MIMICEventPreprocessor):
    def __init__(self, cfg, test=False):
        self.concept_name = 'med'
        super(MIMICPreprocessor_med, self).__init__(cfg, self.concept_name, test)
    @utils.timing_function        
    def __call__(self):
        med = self.load()
        med = self.rename(med)
        super().__call__(med)
        # df = self.handle_range_values(df) we will incorporate it in later preprocessing

    def load(self):
        print('  ::: load PRESCRIPTIONS')
        dose_val_rx_converter = lambda x: x.replace(',','.') if ',' in x else x
        med = pd.read_csv(join(self.raw_data_path, 'PRESCRIPTIONS.csv.gz'), compression='gzip', 
                usecols=['SUBJECT_ID', 'HADM_ID','STARTDATE','ENDDATE', 'DOSE_VAL_RX','DOSE_UNIT_RX','DRUG', 'ICUSTAY_ID'],
                parse_dates=['STARTDATE', 'ENDDATE'], converters={'DOSE_VAL_RX': dose_val_rx_converter},
                dtype=self.dtypes, nrows=self.nrows, infer_datetime_format=True
            ).dropna(subset=['DRUG'])
        return med
    @staticmethod
    def rename(med):
        return med.rename(columns={'SUBJECT_ID': 'PID', 'STARTDATE': 'TIMESTAMP', 'ENDDATE':'TIMESTAMP_END',
            'DRUG': 'CONCEPT', 'DOSE_VAL_RX': 'VALUE', 'DOSE_UNIT_RX': 'VALUE_UNIT', 'HADM_ID': 'ADMISSION_ID'})

    def handle_range_values(self, df):
        """VALUE is often given as range e.g. 1-6, in this case compute mean."""
        df = self.perform_operation_on_column(df, 'VALUE', df.VALUE.str.contains('-'), 
            lambda x: np.average(np.array(x.split('-'), dtype=float)))
        df.VALUE = df.VALUE.astype(float)
        return df
    @staticmethod
    def perform_operation_on_column(df, column, mask, operation):
        """Perform operation on column using loc, with mask and return new column name"""
        df.loc[mask, column] = df.loc[mask, column].map(operation)
        return df
        

class MIMICPreprocessor_pro(MIMICEventPreprocessor):
    def __init__(self, cfg, test=False, concept_name="pro",):
        self.concept_name = concept_name
        super(MIMICPreprocessor_pro, self).__init__(cfg, self.concept_name, test)
        self.concept_dic = {'pro':'PROCEDURES', 'diag':'DIAGNOSES'}
    @utils.timing_function
    def __call__(self):
        pro = self.load()
        adm_dic = self.load_admission_dic()
        pro['TIMESTAMP'] = pro['HADM_ID'].map(adm_dic)
        pro.rename(
            columns={
                'SUBJECT_ID': 'PID',
                'HADM_ID': 'ADMISSION_ID',
                'ICD9_CODE': 'CONCEPT',
                'SEQ_NUM':'VALUE'},
            inplace=True)
        pro['VALUE_UNIT'] = 'SEQ_NUM'
        pro = pro.sort_values(by=['VALUE'])
        super().__call__(pro)

    def load(self):
        print(f'  ::: load {self.concept_dic[self.concept_name]}')
        df = pd.read_csv(join(self.raw_data_path, f'{self.concept_dic[self.concept_name]}_ICD.csv.gz'), compression='gzip', 
                nrows=self.nrows, dtype=self.dtypes).drop(columns=['ROW_ID'])
        return df

    def load_admission_dic(self):
        print('  ::: load ADMISSIONS')
        dfa = pd.read_csv(join(self.raw_data_path, 'ADMISSIONS.csv.gz'), compression='gzip', 
                          parse_dates=['DISCHTIME'],
                          usecols=['HADM_ID', 'DISCHTIME'])
        adm_dic = dfa.set_index('HADM_ID').to_dict()['DISCHTIME']
        return adm_dic


class MIMICPreprocessor_diag(MIMICPreprocessor_pro):
    def __init__(self, cfg, test=False):
        self.concept_name = 'diag'
        super(MIMICPreprocessor_diag, self).__init__(cfg, test, self.concept_name)
        

class MIMICPreprocessor_lab(MIMICEventPreprocessor):
    def __init__(self, cfg, test=False):
        self.concept_name = 'lab'
        super(MIMICPreprocessor_lab, self).__init__(cfg, self.concept_name, test)
    @utils.timing_function
    def __call__(self):
        lab = self.load()
        lab = self.preprocess(lab)
        lab = lab.rename(columns=self.rename_dic)
        super().__call__(lab)

    def load(self):
        print('  ::: load LABEVENTS')
        df = pd.read_csv(join(self.raw_data_path, 'LABEVENTS.csv.gz'),
                         compression='gzip', nrows=self.nrows, parse_dates=['CHARTTIME'],
                         dtype=self.dtypes)
        df = df.drop(columns=['ROW_ID','FLAG', 'VALUENUM'])
        return df

    def preprocess(self, df):
        concept_map = self.get_concept_map()
        df['CONCEPT'] = df.ITEMID.map(concept_map)
        df.drop(columns=['ITEMID', ], inplace=True)
        return df

    def load_dic(self):
        print('  ::: load D_LABITEMS')
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
    @staticmethod
    def process_continuous_values(df_cont):
        df_cont.drop(columns=['VALUE'], inplace=True)
        df_cont.rename(columns={'VALUENUM': 'VALUE'}, inplace=True)
        df_cont['VALUE_CAT'] = 'NaN'
        df_cont = df_cont.loc[df_cont['VALUE'] >= 0].copy()
        return df_cont
    @staticmethod
    def process_categorical_values(df_cat):
        df_cat['VALUE_CAT'] = df_cat['VALUE']
        df_cat['VALUE'] = df_cat.groupby('CONCEPT')['VALUE'].transform(
            lambda x: x.astype('category').cat.codes)
        df_cat.drop(columns=['VALUENUM'], inplace=True)
        df_cat['VALUE_UNIT'] = 'categorical'
        return df_cat
    @staticmethod
    def separate_continuous_categorical(df):
        df_cont = df[df['VALUENUM'].notnull()].copy()
        df_cat = df[df['VALUENUM'].isnull()].copy()
        return df_cont, df_cat

