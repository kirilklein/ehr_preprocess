from ehr_preprocess.preprocessors.base import BasePreprocessor
import pandas as pd
import os

PREPENDS = {'diagnose': 'D', 'medication': 'M', 'procedure': 'P', 'lab': 'L', 'vital': 'V'}

class SyntheaPrepocessor(BasePreprocessor):
    # __init__ is inherited from BasePreprocessor
    def __init__(self, cfg):
        super().__init__(cfg)
        self.logger.info("SyntheaPreprocessor initialized")
        
    def concepts(self):
        # Loop over all top-level concepts (diagnosis, medication, procedures, etc.)
        for type, top_level_config in self.config.concepts.items():
            individual_dfs = [self.load_csv(cfg) for cfg in top_level_config.values()]
            combined = pd.concat(individual_dfs)
            combined = combined.drop_duplicates(subset=['PID', 'CONCEPT', 'TIMESTAMP'])
            combined = combined.sort_values(['PID','TIMESTAMP'])
            if type=='diagnose':
                combined = self.map_snomed_to_icd10(combined)
            combined.CONCEPT = combined.CONCEPT.astype(str)
            combined = self.clean_concepts(combined, type)
            self.save(combined, f'concept.{type}')

    @staticmethod        
    def map_snomed_to_icd10(combined):
        """Map snomed codes to icd10 codes using the provided map"""
        snomed_to_icd10 = SyntheaPrepocessor.get_snomed_to_icd10_mapping()
        combined["new_concept"] = combined.CONCEPT.map(snomed_to_icd10)    
        combined["CONCEPT"] = combined.new_concept.fillna(combined.CONCEPT)
        combined.drop(columns=["new_concept"], inplace=True)
        return combined
    
    @staticmethod
    def get_snomed_to_icd10_mapping():
        """Get the mapping from snomed to icd10 codes"""
        
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct the path to the mapping file
        mapping_file = os.path.join(current_dir, "..", "..", "data", "helper", "tls_Icd10cmHumanReadableMap_US1000124_20230301.tsv")
        
        # Read the mapping file
        map_df = pd.read_csv(mapping_file, sep="\t")
        map_df = map_df.drop_duplicates(subset="referencedComponentId", keep="first")
        map_df.dropna(subset=["mapTarget"], inplace=True)
        snomed_to_icd10 = map_df.set_index("referencedComponentId").mapTarget.to_dict() 
        return snomed_to_icd10
    
    @staticmethod
    def clean_concepts(combined, type):
        """Clean up the concepts"""
        combined.CONCEPT = combined.CONCEPT.str.rstrip("?")
        combined.CONCEPT = combined.CONCEPT.str.replace(".", "", regex=False)
        # add D to the beginning of CONCEPT 
        combined.CONCEPT = PREPENDS[type] + combined.CONCEPT
        combined = combined.dropna(subset=["CONCEPT"])
        return combined



