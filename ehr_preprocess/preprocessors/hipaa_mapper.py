from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException


class HIPAA_Mapper:
    def __init__(self, 
                 driver_exe_path = "C:\\Users\\fjn197\\PhD\\projects\\PHAIR\\pipelines\\ehr_preprocess\\data\\helper\\ndc_to_atc\\chromedriver-win64\\chromedriver.exe",):                 
        self._import()
        self.driver = self.get_driver(driver_exe_path)

    def map_code(self, ndc_code):
        self.enter_code(ndc_code)
        time.sleep(1)
        atc_code = self.read_result()
        return atc_code

    def enter_code(self, ndc_code):
        self.driver.get('https://www.hipaaspace.com/medical-billing/crosswalk-services/ndc-to-atc-mapping/')  # Open the page
        search_field = self.driver.find_element(By.NAME, 'tbxSearchRequest')  # Replace with actual id of search field
        search_field.send_keys(ndc_code)
        search_field.send_keys(Keys.RETURN)

    def get_driver(self, driver_exe_path):
        service = ChromeService(executable_path=driver_exe_path)
        return  webdriver.Chrome(service=service)
    
    def read_result(self):
    # *element = driver.find_element(By.CSS_SELECTOR, 'td table.table.table-bordered a.lookup_item_title')
        try:
            elements = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, 'lookup_item_title'))
            )
            # TODO: check that elements match expression ATCCode - Name
            return elements
        except TimeoutException:
            return None