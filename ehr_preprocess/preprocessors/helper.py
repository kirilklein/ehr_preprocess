from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import time
import pandas as pd

class NDC_ATC_Mapper:
    def __init__(self, raw_ndc_codes, 
                 driver_exe_path = "C:\\Users\\fjn197\\PhD\\projects\\PHAIR\\pipelines\\ehr_preprocess\\data\\helper\\ndc_to_atc\\chromedriver-win64\\chromedriver.exe"):
        self.raw_ndc_codes = raw_ndc_codes
        # self.driver = self.get_driver(driver_exe_path)
        self.all_ndc_codes = self.get_all_ndc_codes() 
        self.possible_formattings = self.get_possible_formattings()
    def __call__(self):
        for raw_ndc_code in self.raw_ndc_codes:
            ndc_code = self.format_ndc_code(raw_ndc_code)
            print(raw_ndc_code, ndc_code)
            yield ndc_code
            # if ndc_code:
            #     atc_code = self.map_ndc_code(ndc_code)
            #     if atc_code:
            #         yield ndc_code, atc_code
            #     else:
            #         yield ndc_code, None            

    def map_code(self, ndc_code):
        if len(ndc_code) < 11:
            return None
        else:
            ndc_code = self.format_ndc_code(ndc_code)
            if ndc_code is None:
                return None
            atc_code = self.retrieve_code(ndc_code)
            return atc_code
        
    def retrieve_code(self, ndc_code):
        self.enter_code(ndc_code)
        time.sleep(1)
        atc_code = self.read_result()
        return atc_code

    def enter_code(self, ndc_code):
        self.driver.get('https://www.hipaaspace.com/medical-billing/crosswalk-services/ndc-to-atc-mapping/')  # Open the page
        search_field = self.driver.find_element(By.NAME, 'tbxSearchRequest')  # Replace with actual id of search field
        search_field.send_keys(ndc_code)
        search_field.send_keys(Keys.RETURN)

    def read_result(self):
    # element = driver.find_element(By.CSS_SELECTOR, 'td table.table.table-bordered a.lookup_item_title')
        try:
            elements = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, 'lookup_item_title'))
            )
            # TODO: check that elements match expression ATCCode - Name
            return elements
        except TimeoutException:
            return None

    def format_ndc_code(self, raw_ndc_code):
        if len(raw_ndc_code)==11:
            formatted_ndc_code = self.insert_separators(raw_ndc_code)
            if formatted_ndc_code not in self.all_ndc_codes:
                formatted_ndc_code = self.move_separator(formatted_ndc_code)
                if formatted_ndc_code not in self.all_ndc_codes:
                    return None
            return formatted_ndc_code
        else:
            return None
    def get_possible_formattings(self,):
        return [self.format1a, self.format1b, self.format2a, self.format2b, self.format3a, self.format3aa, self.format3b, self.format3ba, self.format4a, self.format4b]
    @staticmethod
    def format1a(ndc_code):
        if ndc_code[4] == '0':
            ndc_code = ndc_code[:4] + '-' + ndc_code[5:-2] + '-' + ndc_code[-2:]
        return ndc_code
    @staticmethod
    def format1b(ndc_code):
        if ndc_code[4] == '0':
            ndc_code = ndc_code[:4] + '-' + ndc_code[5:-1] + '-' + ndc_code[-1]
        return ndc_code
    @staticmethod
    def format2a(ndc_code):
        if ndc_code[5] == '0':
            ndc_code = ndc_code[:5] + '-' + ndc_code[6:-2] + '-' + ndc_code[-2:]
        return ndc_code
    @staticmethod
    def format2b(ndc_code):
        if ndc_code[5] == '0':
            ndc_code = ndc_code[:5] + '-' + ndc_code[6:-1] + '-' + ndc_code[-1]
        return ndc_code
    @staticmethod
    def format3a(ndc_code):
        if ndc_code[0] == '0':
            ndc_code = ndc_code[1:5] + '-' + ndc_code[5:-2] + '-' + ndc_code[-2:]
        return ndc_code
    @staticmethod
    def format3aa(ndc_code):
        if ndc_code[0] == '0':
            ndc_code = ndc_code[1:5] + '-' + ndc_code[5:-1] + '-' + ndc_code[-1]
        return ndc_code
    @staticmethod
    def format3b(ndc_code):
        if ndc_code[0] == '0':
            ndc_code = ndc_code[1:6] + '-' + ndc_code[6:-2] + '-' + ndc_code[-2:]
        return ndc_code
    @staticmethod
    def format3ba(ndc_code):
        if ndc_code[0] == '0':
            ndc_code = ndc_code[1:6] + '-' + ndc_code[6:-1] + '-' + ndc_code[-1]
        return ndc_code
    @staticmethod
    def format4a(ndc_code):
        if ndc_code[-2] == '0':
            ndc_code = ndc_code[:4] + '-' + ndc_code[4:-2] + '-' + ndc_code[-1]
        return ndc_code
    @staticmethod
    def format4b(ndc_code):
        if ndc_code[-2] == '0':
            ndc_code = ndc_code[:5] + '-' + ndc_code[5:-2] + '-' + ndc_code[-1]
        return ndc_code
    
    def find_suitable_format(self, raw_code):
        print('raw code', raw_code)
        print('Try exact match')
        formatted_code = self.check_exact_match(raw_code)
        if formatted_code:
            return formatted_code
        print('Try partial match')
        formatted_code = self.check_partial_match(raw_code)
        if formatted_code:
            return formatted_code
        # raise ValueError("No suitable formatting found")
        return None
    
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
        return pd.read_csv("..\\data\\helper\\ndc_to_atc\\NDC_codes.csv").ndcpackagecode
    
    def get_driver(self, driver_exe_path):
        service = ChromeService(executable_path=driver_exe_path)
        return  webdriver.Chrome(service=service)