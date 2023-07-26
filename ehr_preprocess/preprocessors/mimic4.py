from preprocessors import base
import pandas as pd
from os.path import join

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
    
