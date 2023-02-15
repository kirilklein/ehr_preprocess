from ehr_preprocess.processors.base import BasePreprocessor
import pandas as pd


class ComputeromePrepocessor(BasePreprocessor):
    # __init__ is inherited from BasePreprocessor

    def calc_birthdates(self, ages: pd.DataFrame):
        if self.config.patients_info.ages.get('covid_tests') is not None:
            def fill_with_covid_tests(self, ages: pd.DataFrame):
                tests = self.load_csv(self.config.patients_info.ages.covid_tests).drop_duplicates('ADMISSION_ID')
                test_dict = {k: v for k,v in tests.values}  # Create dict of (ADMISSION_ID: timestamp)
                nan_ages = ages[ages['TIMESTAMP'].isna()]   # Get rows with missing timestamp
                new_timestamps = nan_ages['PID'].map(lambda key: test_dict.get(key))    # Map PID to covid test timestamp
                ages.loc[new_timestamps.index, 'TIMESTAMP'] = new_timestamps.values           # Overwrite NaN values with covid test timestamps

                return ages
            ages = fill_with_covid_tests(ages)

        # Calculate approximate birthdates
        ages = ages.dropna().drop_duplicates('PID', keep='last')    # Drop rows with missing timestamp or duplicate patient ids
        ages['BIRTHDATE'] = ages['TIMESTAMP'] - ages['AGE'].map(lambda years: pd.Timedelta(years*365.25, 'D'))  # Calculate an approximate birthdate from their current age

        return ages[['PID', 'BIRTHDATE']]

    def update_with_admission_info(self, df: pd.DataFrame, pid: bool = True, timestamp: bool = True):
        def extract_key(admission_id, key):
            if admission_id in self.admission_info:
                return self.admission_info[admission_id][key]
            else:
                return None
        if pid:
            df['PID'] = df['ADMISSION_ID'].map(lambda x: extract_key(x, 'PID'))
        if timestamp:
            df['TIMESTAMP'] = df['ADMISSION_ID'].map(lambda x: extract_key(x, 'TIMESTAMP'))

        return df

