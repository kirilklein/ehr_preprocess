
import time
import os
import numpy as np
from os.path import join
import inspect
import pandas as pd
import glob


class NDC_ATC_Mapper:
    def __init__(self, medications, 
                 test=True):
        self.medications = medications
        self.raw_ndc_codes = medications['CONCEPT'].unique()
        
        self.test = test

    def map(self):
        # self.format_ndc_codes()
        mapping_ = self.get_mapping()
            
        
    def get_mapping(self):
        store_dir = "..\\data\\interim\\mimic4"
        files = glob.glob(join(store_dir, "ndc_map*.csv"))
        if len(files)>1:
            raise ValueError(f'Multiple files found:{files}')
        elif len(files)==0:
            self.medications.rename(columns={'CONCEPT': 'NDC'})['NDC'].to_csv(join(store_dir, "ndc_codes.csv"), index=False)
            raise FileNotFoundError(f"Run https://github.com/fabkury/ndc_map on the NDC codes stored in {store_dir} to create the mapping file.\
                             and save the output in {store_dir}. Then Re-run this script.")
        else:
            return pd.read_csv(files[0], usecols=['ndc', 'atc5'], dtype={'ndc': 'category', 'atc5': 'category'})

    def get_random_atc5(ndc_code):
        atc5_codes = mapping_[mapping_['NDC'] == ndc]['atc5']
        return np.random.choice(atc5_codes)
    def format_ndc_codes(self):
        formatter = NDC_Formatter(self.raw_ndc_codes, test=self.test)
        formatter.format()
        self.formatted_codes = formatter.format_map
        self.inverse_format_map = formatter.get_inverse_format_map()

    def map_code(self, ndc_code):
        if len(ndc_code) < 11:
            return None
        else:
            ndc_code = self.format_ndc_code(ndc_code)
            if ndc_code is None:
                return None
            atc_code = self.retrieve_code(ndc_code)
            return atc_code
        
class NDC_Formatter:
    def __init__(self, raw_ndc_codes, test=True):
        self.raw_ndc_codes = raw_ndc_codes
        self.all_ndc_codes = self.get_all_ndc_codes() 
        self.possible_formattings = self.get_possible_formattings()
        self.format_map = {}
        self.test = test

    def format(self):
        self.format_codes()
        self.save_formatted_codes()
        if not os.path.exists('..\\data\\helper\\ndc_to_atc\\NDC_to_ATC_mapping.csv'):
            raise ValueError('Run https://github.com/fabkury/ndc_map on the formatted NDC codes to create the mapping file.\
                             and save it in ..\\data\\helper\\ndc_to_atc\\NDC_to_ATC_mapping.csv')
    def save_formatted_codes(self):
        data = [(k, v) for k, values in self.format_map.items() for v in values]
        formatted_codes = pd.DataFrame(data, columns=['Raw_NDC', 'NDC'])
        os.makedirs('..\\data\\interim\\mimic4', exist_ok=True)
        formatted_codes.to_csv('..\\data\\interim\\mimic4\\formatted_ndc_codes.csv', index=False)

    def get_inverse_format_map(self):
        return {v: k for k, ls in self.format_map.items() for v in ls if v is not None}
    def format_codes(self):
        for raw_code in self.raw_ndc_codes:
            if len(raw_code) < 11:
                self.format_map[raw_code] = [None]
            else:
                self.find_suitable_format(raw_code)

    def get_possible_formattings(self,):
        def _format1a(ndc_code):
            if ndc_code[4] == '0':
                ndc_code = ndc_code[0:4] + '-' + ndc_code[5:-2] + '-' + ndc_code[-2:]
            return ndc_code
        def _format1b(ndc_code):
            if ndc_code[4] == '0':
                ndc_code = ndc_code[0:4] + '-' + ndc_code[5:-1] + '-' + ndc_code[-1:]
            return ndc_code
        def _format2a(ndc_code):
            if ndc_code[5] == '0':
                ndc_code = ndc_code[0:5] + '-' + ndc_code[6:-2] + '-' + ndc_code[-2:]
            return ndc_code
        def _format2b(ndc_code):
            if ndc_code[5] == '0':
                ndc_code = ndc_code[0:5] + '-' + ndc_code[6:-1] + '-' + ndc_code[-1:]
            return ndc_code
        def _format3a(ndc_code):
            if ndc_code[0] == '0':
                ndc_code = ndc_code[1:5] + '-' + ndc_code[5:-2] + '-' + ndc_code[-2:]
            return ndc_code
        def _format3aa(ndc_code):
            if ndc_code[0] == '0':
                ndc_code = ndc_code[1:5] + '-' + ndc_code[5:-1] + '-' + ndc_code[-1:]
            return ndc_code
        def _format3b(ndc_code):
            if ndc_code[0] == '0':
                ndc_code = ndc_code[1:6] + '-' + ndc_code[6:-2] + '-' + ndc_code[-2:]
            return ndc_code
        def _format3ba(ndc_code):
            if ndc_code[0] == '0':
                ndc_code = ndc_code[1:6] + '-' + ndc_code[6:-1] + '-' + ndc_code[-1:]
            return ndc_code
        def _format4a(ndc_code):
            if ndc_code[-2] == '0':
                ndc_code = ndc_code[0:4] + '-' + ndc_code[4:-2] + '-' + ndc_code[-1:]
            return ndc_code
        def _format4b(ndc_code):
            if ndc_code[-2] == '0':
                ndc_code = ndc_code[0:5] + '-' + ndc_code[5:-2] + '-' + ndc_code[-1:]
            return ndc_code
        return [method for name, method in locals().items() if name.startswith('_format')]
    
    def find_suitable_format(self, raw_code):
        formatted_code = self.check_exact_match(raw_code)
        if formatted_code:
            self.format_map[raw_code] = [formatted_code]
        else:
            formatted_code = self.check_partial_match(raw_code)
            if formatted_code:
                self.format_map[raw_code] = [formatted_code]
            else:
                self.format_map[raw_code] = [self.apply_format(raw_code, formatting) for formatting in self.possible_formattings]
                self.format_map[raw_code] = [code for code in self.format_map[raw_code] if code!=raw_code]
    
    def check_exact_match(self, raw_code):
        for formatting in self.possible_formattings:
            try:
                formatted_code = self.apply_format(raw_code, formatting)
                if self.check_format_exact(formatted_code):
                    return formatted_code
            except Exception:
                continue

    def check_partial_match(self, raw_code):
        for formatting in self.possible_formattings:
            try:
                formatted_code = self.apply_format(raw_code, formatting)
                if self.check_format_partial(formatted_code):
                    return formatted_code
            except Exception:
                continue

    def apply_format(self, raw_code, formatting):
        return formatting(raw_code)
    
    def check_format_exact(self, ndc_code):
        return ndc_code in self.all_ndc_codes.values
    
    def check_format_partial(self, ndc_code):
        first_nine = self.all_ndc_codes.str.startswith(ndc_code[:9]).any()
        firs_eight = self.all_ndc_codes.str.startswith(ndc_code[:8]).any()
        first_seven = self.all_ndc_codes.str.startswith(ndc_code[:7]).any()
        return first_nine or firs_eight or first_seven
    @staticmethod
    def get_all_ndc_codes():
        ndc1 = pd.read_csv("..\\data\\helper\\ndc_to_atc\\ndc2atc_level4.csv").NDC
        ndc2 = pd.read_csv("..\\data\\helper\\ndc_to_atc\\NDC_codes.csv").ndcpackagecode
        return pd.concat([ndc1, ndc2]).drop_duplicates()