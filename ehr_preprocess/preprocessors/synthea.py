from preprocessors.base import BasePreprocessor
import pandas as pd
import numpy as np

def check_dose(index, words):
    if index+1 >= len(words):
        return False
    elif words[index][0] in "0123456789" and words[index][-1] in "0123456789" and words[index+1] != '%':
        return True
    else:
        return False

class SyntheaPrepocessor(BasePreprocessor):
    # __init__ is inherited from BasePreprocessor

    def concepts(self):
        # Loop over all top-level concepts (diagnosis, medication, procedures, etc.)
        for type, top_level_config in self.config.concepts.items():
            individual_dfs = [self.load_csv(cfg) for cfg in top_level_config.values()]
            combined = pd.concat(individual_dfs)
            combined = combined.drop_duplicates(subset=['PID', 'CONCEPT', 'TIMESTAMP'])
            # combined = combined.sort_values('TIMESTAMP')
            if type == "diagnose":
                combined = self.map_snomed_to_icd10(combined)
                combined = self.clean_concepts(combined)
            elif type == "medication":
                combined = self.clean_medications(combined)
            combined = combined.sort_values('TIMESTAMP')
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


    @staticmethod
    def clean_medications(combined):
        """Clean up the medications, add M to the beginning of CONCEPT, and split medicine name, dose, and unit"""
        # add M to the beginning of CONCEPT
        combined.CONCEPT = combined.CONCEPT.astype(str)
        combined.CONCEPT = "M" + combined.CONCEPT
        combined = combined.dropna(subset=["CONCEPT"])
        # add columns for medicine name, dose, unit, and method
        combined["MEDICINE"] = np.nan
        combined["DOSE"] = np.nan
        combined["UNIT"] = np.nan
        combined["METHOD"] = np.nan

        df_temp = pd.DataFrame(columns=combined.columns)

        # split medicine name, dose, and unit
        for i in combined.index:
            if i % 10000 == 0:
                print("Processing row " + str(i) + " of " + str(len(combined)))
            medication = combined.loc[i, "MEDICATION"]
            timestamp = combined.loc[i, "TIMESTAMP"]
            pid = combined.loc[i, "PID"]
            admission_id = combined.loc[i, "ADMISSION_ID"]
            concept = combined.loc[i, "CONCEPT"]

            ms_in_medication = medication.split(" / ")
            # if there is only one medicine in one medication record
            if len(ms_in_medication) == 1:
                words = ms_in_medication[0].split()
                numbers = [n for n in range(len(words)) if check_dose(n, words)]
                # if there is no dose, unit and method
                if len(numbers) == 0:
                    medicine = ms_in_medication[0]
                    dose = ""
                    unit = ""
                    method = ""
                else:
                    medicine = " ".join(words[:numbers[-1]])
                    dose = words[numbers[-1]]
                    unit = words[numbers[-1]+1]
                    method = " ".join(words[numbers[-1]+2:])
                # add the splited medicine name, dose, unit, and method to the combined dataframe
                combined.loc[i, "MEDICINE"] = medicine
                combined.loc[i, "DOSE"] = dose
                combined.loc[i, "UNIT"] = unit
                combined.loc[i, "METHOD"] = method
            
            # if there are multiple medicines in one medication record
            # split the entry into multiple entries
            else:
                medicine_s = []
                dose_s = []
                unit_s = []
                method = ""
                
                for k in range(len(ms_in_medication)):
                    # if it is the first medicine, 
                    if k == 0:
                        medication_k = ms_in_medication[k].split()
                        numbers_k = [n for n in range(len(medication_k)) if check_dose(n, medication_k)]
                        if len(numbers_k) == 0:
                            medicine = " ".join(medication_k)
                            dose = ""
                            unit = ""
                        else:
                            medicine = " ".join(medication_k[:numbers_k[-1]])
                            dose = medication_k[numbers_k[-1]]
                            unit = medication_k[numbers_k[-1]+1]
                        medicine_s.append(medicine)
                        dose_s.append(dose)
                        unit_s.append(unit)
                    # only the last medicine has method
                    elif k == len(ms_in_medication) - 1:
                        medication_k = ms_in_medication[k].split()
                        numbers_k = [n for n in range(len(medication_k)) if check_dose(n, medication_k)]
                        if len(numbers_k) == 0:
                            medicine = " ".join(medication_k)
                            dose = ""
                            unit = ""
                        else:
                            medicine = " ".join(medication_k[:numbers_k[-1]])
                            dose = medication_k[numbers_k[-1]]
                            unit = medication_k[numbers_k[-1]+1]
                            method_l = " ".join(medication_k[numbers_k[-1]+2:])
                        medicine_s.append(medicine)
                        dose_s.append(dose)
                        unit_s.append(unit)
                        method += method_l
                    else:
                        medication_k = ms_in_medication[k].split()
                        numbers_k = [n for n in range(len(medication_k)) if check_dose(n, medication_k)]
                        if len(numbers_k) == 0:
                            medicine = " ".join(medication_k)
                            dose = ""
                            unit = ""
                        else:
                            medicine = " ".join(medication_k[:numbers_k[-1]])
                            dose = medication_k[numbers_k[-1]]
                            unit = medication_k[numbers_k[-1]+1]
                        medicine_s.append(medicine)
                        dose_s.append(dose)
                        unit_s.append(unit)
                        
                # add the first of splited medicine name, dose, unit, and method to the combined dataframe
                combined.loc[i, "MEDICINE"] = medicine_s[0]
                combined.loc[i, "DOSE"] = dose_s[0]
                combined.loc[i, "UNIT"] = unit_s[0]
                combined.loc[i, "METHOD"] = method
                # add the rest of splited medicine name, dose, unit, and method to the temporary dataframe
                for j in range(1, len(medicine_s)):
                    df_temp.loc[len(df_temp)] = [timestamp, pid, admission_id, concept, medication, medicine_s[j], dose_s[j], unit_s[j], method]
         
        # add the temporary dataframe to the combined dataframe
        combined = pd.concat([combined, df_temp], ignore_index=True)
        return combined