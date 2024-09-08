import pandas as pd
from ehr_preprocess.preprocessors.mimic_helper import NDC_ATC_Mapper
from ehr_preprocess.preprocessors import base


class MIMIC4Preprocessor(base.BasePreprocessor):
    """Extracts events from MIMIC-III database and saves them in a single file."""

    def __init__(self, cfg):
        super(MIMIC4Preprocessor, self).__init__(cfg)

    def processs(self):
        self.patients_info()
        self.concepts()
    
    def concepts(self):
        # Loop over all top-level concepts (diagnosis, medication, procedures, etc.)
        for type, top_level_config in self.config.concepts.items():
            print(f'Processing {type}...')
            individual_dfs = [self.load_csv(cfg) for cfg in top_level_config.values()]
            combined = self.combine_dataframes(individual_dfs, top_level_config)
            
            combined = combined.drop_duplicates(subset=['PID', 'CONCEPT', 'TIMESTAMP'])
            combined = combined.dropna(subset=['PID', 'CONCEPT','TIMESTAMP'], how='any')
            self.save(combined, f'concept.{type}')

    def combine_dataframes(self, individual_dfs, top_level_config):
        first_config = [cfg for cfg in top_level_config.values()][0]
        if 'combine' not in first_config:
            return pd.concat(individual_dfs)
        else:
            combine = first_config.combine.get('method', 'concat')
            if combine=='concat':
                combined = pd.concat(individual_dfs)
            elif combine=='merge':
                merge_on = first_config.combine.get('on', None)
                combined = individual_dfs[0]
                for df in individual_dfs[1:]:
                    combined = combined.merge(df, on=merge_on, how='left')
            else:
                raise ValueError(f'Unknown combine method: {combine}')
            return combined

    def calculate_birthdate(self, patients):
        """Calculates the birthdate of each patient based on the anchor year and age."""
        patients['DATE_OF_BIRTH'] = patients['anchor_year'] - patients['anchor_age']
        patients = patients.drop(columns=['anchor_year', 'anchor_age'])
        patients['DATE_OF_BIRTH'] = pd.to_datetime(patients['DATE_OF_BIRTH'], format='%Y')
        return patients
    
    def majority_vote(self, admissions):
        """Returns the most common value for each column in the admissions dataframe."""
        admission_info = pd.DataFrame(admissions['PID'].unique(), columns=['PID'])
        for col in admissions.columns[1:]:
            col_info = admissions.groupby('PID')[col].agg(lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else None).reset_index()
            admission_info = admission_info.merge(col_info, on='PID', how='left')
        return admission_info
    
    def map_icd9_to_icd10(self, diagnoses):
        mapping = self.get_icd9_to_icd10_mapping()
        icd9_mask = diagnoses['icd_version']==9
        diagnoses = diagnoses.drop(columns=['icd_version'])
        
        diagnoses.loc[icd9_mask, 'CONCEPT'] = diagnoses.loc[icd9_mask, 'CONCEPT'].map(mapping)
        return diagnoses

    def get_icd9_to_icd10_mapping(self):
        mapping = pd.read_csv("..\\data\\helper\\diagnosis_gems_2018\\2018_I9gem.txt", delimiter='\s+', names=['icd9', 'icd10'], usecols=[0,1])
        mapping['icd10'] = 'D' + mapping['icd10']
        mapping['icd9'] = 'D' + mapping['icd9']
        mapping = mapping.set_index('icd9')['icd10'].to_dict()
        return mapping
    
    def handle_medication(self, medication, prepend='M'):
        medication = PrescriptionMedicationHandler()(medication, prepend)
        return medication
    
    
class PrescriptionMedicationHandler:
    """Maps NDC codes to ATC5 codes. And fills nans"""
    def __call__(self, medication, prepend) -> pd.DataFrame:
        medication = self.map_ndc_to_atc(medication)
        medication = self.fill_nans_medication(medication)
        medication = self._prepend(medication, prepend)
        return medication
    
    @staticmethod
    def map_ndc_to_atc(medication):
        mapper_ = NDC_ATC_Mapper(medication)
        return mapper_.map()
    @staticmethod
    def fill_nans_medication(medication):
        medication.CONCEPT = medication.CONCEPT.fillna(medication.drug)
        medication = medication.drop(columns=['drug'])
        return medication
    @staticmethod
    def _prepend(df, prepend):
        df['CONCEPT'] = prepend + df['CONCEPT']
        return df

    
    