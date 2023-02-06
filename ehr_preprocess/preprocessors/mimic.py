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
        if os.path.exists(join(self.processed_data_path, 'metadata.json')):
            with open(join(self.processed_data_path, 'metadata.json'), 'r') as fp:
                self.metadata_ls = json.load(fp)
        else:
            self.metadata_ls = []

    def save_metadata(self):
        with open(join(self.processed_data_path, 'metadata.json'), 'w') as fp:
            json.dump(self.metadata_ls, fp)

    def update_metadata(self, type, coding_system,
                        file, prepend, src_files_ls):
        concept_dic = {
            'Type': type, 'Coding_System': coding_system, 'File': file, 'Prepend': prepend, 'Source': src_files_ls
        }
        if concept_dic not in self.metadata_ls:
            self.metadata_ls.append(concept_dic)


class MIMIC3Preprocessor(BasePreprocessor):
    """Extracts events from MIMIC-III database and saves them in a single file."""

    def __init__(self, cfg, test=False):
        super(MIMIC3Preprocessor, self).__init__(cfg, test)

    def __call__(self):
        print('Preprocess Mimic3 from: ', self.raw_data_path)
        print(":Extract patient info")
        self.extract_patient_info()
        print(":Extract events")
        for concept_name in self.cfg.concepts:
            save_path = join(
                self.processed_data_path,
                f'concept.{concept_name}.parquet')
            extractor = getattr(self, f"extract_{concept_name}")
            if os.path.exists(save_path):
                if utils.query_yes_no(
                        f"File {save_path} already exists. Overwrite?"):
                    extractor(concept_name)
                else:
                    print(f"Skipping {concept_name}")
            else:
                extractor(concept_name)
        print(":Save metadata")
        self.save_metadata()

    def extract_patient_info(self):
        pass

    def extract_diag(self, concept_name):
        print("::Extract diagnoses")
        MIMICDiagPreprocessor(self.cfg, self.test)(concept_name)
        self.update_metadata(
            'Diag', 'ICD9', f'concept.{concept_name}.parquet', 'D', [
                'DIAGNOSES_ICD.csv.gz', 'ADMISSIONS.csv.gz'])

    def extract_med(self, concept_name):
        print("::Extract medications")
        MIMICMedPreprocessor(self.cfg, self.test)(concept_name)
        self.update_metadata(
            'Med', 'DrugName', f'concept.{concept_name}.parquet', 'M', [
                'PRESCRIPTIONS.csv.gz'])

    def extract_pro(self, concept_name):
        pass

    def extract_lab(self, concept_name):
        print("::Extract lab results")
        MIMICLabPreprocessor(self.cfg, self.test)(concept_name)
        self.update_metadata(
            'Lab', 'LOINC', f'concept.{concept_name}.parquet', 'L', [
                'LABEVENTS.csv.gz', 'D_LABITEMS.csv.gz'])


class MIMICMedPreprocessor(MIMIC3Preprocessor):
    def __init__(self, cfg, test=False):
        super(MIMICMedPreprocessor, self).__init__(cfg, test)

    def __call__(self, concept_name):
        df = self.load()
        df = self.rename(df)
        df = self.handle_range_values(df)
        df['CONCEPT'] = df['CONCEPT'].map(lambda x: 'M' + str(x))
        df.to_parquet(
            join(self.processed_data_path, f'concept.{concept_name}.parquet'), index=False)

    def load(self):
        df = pd.read_csv(join(self.raw_data_path, 'PRESCRIPTIONS.csv.gz'), compression='gzip', nrows=self.nrows,
                         usecols=[
            'SUBJECT_ID',
            'HADM_ID',
            'STARTDATE',
            'DOSE_VAL_RX',
            'DOSE_UNIT_RX',
            'DRUG'],
            parse_dates=['STARTDATE']).dropna(subset=['DRUG'])
        return df

    def rename(self, df):
        return df.rename(columns={'SUBJECT_ID': 'PID', 'STARTDATE': 'TIMESTAMP', 'DRUG': 'CONCEPT',
                                  'DOSE_VAL_RX': 'VALUE', 'DOSE_UNIT_RX': 'UNIT_VALUE', 'HADM_ID': 'ADMISSION_ID'})

    def handle_range_values(self, df):
        """VALUE is often given as range e.g. 1-6, in this case compute mean.
          replace , with . in VALUE"""
        df = self.perform_operation_on_column(df, 'VALUE', df.VALUE.str.contains('-'), 
            lambda x: np.average(np.array(x.split('-'), dtype=float)))
        df.VALUE = df.VALUE.astype(str)
        df = self.perform_operation_on_column(df, 'VALUE', df.VALUE.str.contains(','), 
            lambda x: x.replace(',', '.'))
        df.VALUE = df.VALUE.astype(float)
        return df

    def perform_operation_on_column(self, df, column, mask, operation):
        """Perform operation on column using loc, with mask and return new column name"""
        df.loc[mask, column] = df.loc[mask, column].map(operation)
        return df
        

class MIMICDiagPreprocessor(MIMIC3Preprocessor):
    def __init__(self, cfg, test=False):
        super(MIMICDiagPreprocessor, self).__init__(cfg, test)

    def __call__(self, concept_name):
        df = self.load()
        adm_dic = self.load_admission_dic()
        df['TIMESTAMP'] = df['HADM_ID'].map(adm_dic)
        df.rename(
            columns={
                'SUBJECT_ID': 'PID',
                'HADM_ID': 'ADMISSION_ID',
                'ICD9_CODE': 'CONCEPT'},
            inplace=True)
        df['CONCEPT'] = df['CONCEPT'].map(lambda x: 'D' + str(x))
        df.to_parquet(
            join(
                self.processed_data_path,
                f'concept.{concept_name}.parquet'),
            index=False)

    def load(self):
        df = pd.read_csv(join(self.raw_data_path, 'DIAGNOSES_ICD.csv.gz'), compression='gzip', nrows=self.nrows,
                         dtype={'SEQ_NUM': 'Int32'}).drop(columns=['ROW_ID'])
        return df

    def load_admission_dic(self):
        dfa = pd.read_csv(join(self.raw_data_path, 'ADMISSIONS.csv.gz'), compression='gzip', nrows=self.nrows,
                          parse_dates=['ADMITTIME'],
                          usecols=['HADM_ID', 'ADMITTIME'])
        adm_dic = dfa.set_index('HADM_ID').to_dict()['ADMITTIME']
        return adm_dic


class MIMICLabPreprocessor(MIMIC3Preprocessor):

    def __init__(self, cfg, test=False):
        super(MIMICLabPreprocessor, self).__init__(cfg, test)

    def __call__(self, concept_name):
        df = self.load()
        df_dic = self.load_dic()
        df = self.preprocess(df, df_dic)
        df.to_parquet(
            join(
                self.processed_data_path,
                f'concept.{concept_name}.parquet'),
            index=False)

    def load(self):
        df = pd.read_csv(join(self.raw_data_path, 'LABEVENTS.csv.gz'),
                         compression='gzip', nrows=self.nrows, parse_dates=['CHARTTIME'],
                         dtype={'SUBJECT_ID': 'Int32', 'ITEMID': 'Int32', 'VALUE': 'str', 'VALUENUM': 'float32', 'VALUEUOM': 'str', 'HADM_ID': 'Int32'})
        df = df.rename(
            columns={
                'SUBJECT_ID': 'PID',
                'CHARTTIME': 'TIMESTAMP',
                'VALUEUOM': 'VALUE_UNIT',
                'HADM_ID': 'ADMISSION_ID'}).drop(
            columns=[
                'ROW_ID',
                'FLAG'])
        return df

    def load_dic(self):
        return pd.read_csv(
            join(self.raw_data_path, 'D_LABITEMS.csv.gz'), compression='gzip')

    def preprocess_dic(self, df_dic):
        # add more specification to SPECIMEN using the fluid column
        df_dic.loc[df_dic['LABEL'].str.contains('SPECIMEN'), 'LABEL'] \
            = df_dic.loc[df_dic['LABEL'].str.contains('SPECIMEN'), 'FLUID']\
            + ' ' + \
            df_dic.loc[df_dic['LABEL'].str.contains('SPECIMEN'), 'CATEGORY']
        return df_dic

    def preprocess(self, df, df_dic):
        concept_map = self.get_concept_map(df_dic)
        df['CONCEPT'] = df.ITEMID.map(concept_map)
        df.drop(columns=['ITEMID'], inplace=True)
        df = self.process_values(df)
        return df

    def get_concept_map(self, df_dic):
        item_code_dic = pd.Series(
            df_dic.LOINC_CODE.values,
            index=df_dic.ITEMID).to_dict()
        item_name_dic = pd.Series(df_dic[df_dic.LOINC_CODE.isna(
        )].LABEL.values, index=df_dic[df_dic.LOINC_CODE.isna()].ITEMID).to_dict()
        # combine dicts
        return {**item_code_dic, **item_name_dic}

    def process_values(self, df):
        df_cont, df_cat = self.separate_continuous_categorical(df)
        df_cont = self.process_continuous_values(df_cont)
        df_cat = self.process_categorical_values(df_cat)
        df = pd.concat([df_cont, df_cat])
        # sort by PID and ADMISSION_ID
        df.sort_values(by=['PID', 'ADMISSION_ID', 'TIMESTAMP'], inplace=True)
        df.CONCEPT = df.CONCEPT.map(lambda x: 'L' + x)
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
