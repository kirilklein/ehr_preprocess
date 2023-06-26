from preprocessors.base import BasePreprocessor
import pandas as pd

class SyntheaPrepocessor(BasePreprocessor):
    # __init__ is inherited from BasePreprocessor

    def concepts(self):
        # Loop over all top-level concepts (diagnosis, medication, procedures, etc.)
        for type, top_level_config in self.config.concepts.items():
            individual_dfs = [self.load_csv(cfg) for cfg in top_level_config.values()]
            combined = pd.concat(individual_dfs)
            combined = combined.drop_duplicates(subset=['PID', 'CONCEPT', 'TIMESTAMP'])
            combined = combined.sort_values('TIMESTAMP')
            combined = self.map_snomed_to_icd10(combined)
            combined = self.clean_concepts(combined)
            self.save(combined, f'concept.{type}')

    @staticmethod        
    def map_snomed_to_icd10(combined):
        """Map snomed codes to icd10 codes using the provided map"""
        map_df = pd.read_csv("helpers\\tls_Icd10cmHumanReadableMap_US1000124_20230301.tsv", sep="\t", )
        map_df = map_df.drop_duplicates(subset="referencedComponentId", keep="first")
        snomed_to_icd10 = map_df.set_index("referencedComponentId").mapTarget.to_dict() 
        combined.CONCEPT = combined.CONCEPT.map(snomed_to_icd10)
        return combined
    
    @staticmethod
    def clean_concepts(combined):
        """Clean up the concepts"""
        combined.CONCEPT = combined.CONCEPT.str.rstrip("?")
        combined.CONCEPT = combined.CONCEPT.str.replace(".", "")
        # add D to the beginning of CONCEPT 
        combined.CONCEPT = "D" + combined.CONCEPT
        combined = combined.dropna(subset=["CONCEPT"])
        return combined



