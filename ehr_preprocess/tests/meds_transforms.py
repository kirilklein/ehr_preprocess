import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from ehr_preprocess.preprocessors.meds_transforms import transform_admission_discharge

# Assuming transform_admission_discharge function is already defined

class TestTransformAdmissionDischarge(unittest.TestCase):
    
    def setUp(self):
        # Sample data to test the function
        self.data = {
            'subject_id': [1, 1, 1, 1, 2, 2],
            'time': ['2024-01-01 10:00', '2024-01-01 11:00', '2024-01-01 12:00', '2024-01-01 14:00', '2024-01-02 09:00', '2024-01-02 17:00'],
            'admission_id': ['adm1', 'adm1' ,'adm1', 'adm2', 'adm3', 'adm3'],
            'code': ['A', 'B', 'C', 'D', 'E']
        }
        self.df = pd.DataFrame(self.data)
        self.df['time'] = pd.to_datetime(self.df['time'])  # Convert 'time' to datetime format

    def test_transform_admission_discharge(self):
        # Expected result after transforming admission and discharge events
        expected_data = {
            'subject_id': [1, 1, 1, 2, 2],
            'time': ['2024-01-01 10:00', '2024-01-01 12:00', '2024-01-01 14:00', '2024-01-01 14:00', '2024-01-02 09:00', '2024-01-02 17:00'],
            'code': ['Admission', 'Discharge', 'Admission', 'Discharge', 'Admission', 'Discharge']
        }
        expected_df = pd.DataFrame(expected_data)
        expected_df['time'] = pd.to_datetime(expected_df['time'])  # Convert 'time' to datetime format
        expected_df = expected_df.sort_values(by=['subject_id', 'time']).reset_index(drop=True)
        
        # Call the function
        result_df = transform_admission_discharge(self.df)

        # Assert that the result matches the expected DataFrame
        assert_frame_equal(result_df.reset_index(drop=True), expected_df)

if __name__ == '__main__':
    unittest.main()
