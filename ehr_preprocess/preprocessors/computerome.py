from ehr_preprocess.preprocessors.base import BasePreprocessor
import pandas as pd


class ComputeromePrepocessor(BasePreprocessor):
    # __init__ is inherited from BasePreprocessor

    # Load concepts then create patients and events
    def concepts(self):
        concepts_info = self.config.concepts

        for key, info in concepts_info.items():
            df = self.load_csv(info)
            df = self.format(df)
            df = df.sort_values('TIMESTAMP')

            self.to_parquet(df, f'concept.{key}.parquet')

    def patient_info(self):

        # Update patients with birthdates
        self.set_birthdates()

        # Update patients with demographics
        self.set_demographics()

        # Convert info dict to dataframe
        df = self.info_to_df()

        self.to_parquet(df, 'patients_info.parquet')

    def set_demographics(self):
        demo = self.load_csv(self.config.demographics)

        demo.apply(lambda patient: 
            self.info.get(patient['PID'])
            .update(
                GENDER=patient['GENDER'], 
                WEIGHT=patient['WEIGHT'], 
                HEIGHT=patient['HEIGHT'], 
                BMI=patient['BMI']
            ), axis=1
        )
        return demo

    def set_birthdates(self):
        ages_info = self.config.ages
        ages = self.load_csv(ages_info)

        if 'covid_tests' in ages_info:
            ages = self.fill_with_covid_tests(ages)

        # Calculate approximate birthdates
        ages = ages.dropna().drop_duplicates('PID', keep='last')    # Drop rows with missing timestamp or duplicate patient ids
        ages['BIRTHDATE'] = ages['TIMESTAMP'] - ages['AGE'].map(lambda years: pd.Timedelta(years*365.25, 'D'))  # Calculate an approximate birthdate from their current age

        # Update patients with birthdates
        ages.apply(lambda patient:
            self.info.get(patient['PID'])
            .update(
                BIRTHDATE=patient['BIRTHDATE'],
            ), axis=1
        )
        return ages

    def fill_with_covid_tests(self, ages: pd.DataFrame):
        tests = self.load_csv(self.config.covid_tests).drop_duplicates('ADMISSION_ID')
        test_dict = {k: v for k,v in tests.values}  # Create dict of (ADMISSION_ID: timestamp)
        nan_ages = ages[ages['TIMESTAMP'].isna()]   # Get rows with missing timestamp
        new_timestamps = nan_ages['PID'].map(lambda key: test_dict.get(key))    # Map PID to covid test timestamp
        ages.loc[new_timestamps, 'TIMESTAMP'] = new_timestamps.values           # Overwrite NaN values with covid test timestamps

        return ages

