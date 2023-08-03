import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import dirname, join, realpath

class StatisticMedication():
    def __init__(self, filepaths):
        self.filepaths = filepaths
    
    def __call__(self):
        med_dict_all = {}
        for filepath in self.filepaths:
            med_dict = self.get_med_dict(filepath)
            med_dict_all = self.merge_two_med_dict(med_dict_all, med_dict)
        
        lengths = self.stat_unique_values_for_one_medicine(med_dict_all)
        print("The number of unique medicine names: ", len(med_dict_all.keys()))
        print("The number of unique dose-unit combinations: ", self.stat_unique_values_in_total(med_dict_all))
        print("The mean of unique dose-unit combinations for one medicine: ", np.mean(lengths))
        print("The std of unique dose-unit combinations for one medicine: ", np.std(lengths))

        plt.hist(lengths, bins=100)
        plt.xlabel("The number of unique dose-unit combinations for one medicine")
        plt.ylabel("The number of medicines")
        plt.title("The number of unique dose-unit combinations for one medicine")
        plt.savefig('histogram.png')
        plt.show()

    @staticmethod
    def get_med_dict(filepath): 
        # load data in chunks
        med_dict = {}
        for chunk in pd.read_csv(filepath, chunksize=1000000):
            # drop duplicates
            chunk = chunk.drop_duplicates(subset=['PID', 'CONCEPT', 'TIMESTAMP'])
            for idx in chunk.index:
                medicine_name = chunk.loc[idx, 'MEDICINE']
                dose = str(chunk.loc[idx, 'DOSE'])
                unit = str(chunk.loc[idx, 'UNIT'])
                if medicine_name not in med_dict.keys():
                    med_dict[medicine_name] = [(dose, unit)]
                else:
                    if (dose, unit) not in med_dict[medicine_name]:
                        med_dict[medicine_name].append((dose, unit))
        return med_dict
    
    @staticmethod
    def stat_unique_values_for_one_medicine(med_dict):
        # statistics for unique values for one medication
        lengths = []
        for medicine, dose_unit_list in med_dict.items():
            lengths.append(len(dose_unit_list))
        return lengths
    
    @staticmethod
    def stat_unique_values_in_total(med_dict):
        # statistics for unique values in total(in combination with the unit)
        dose_unit_lists = []
        for medicine, dose_unit_list in med_dict.items():
            dose_unit_lists += dose_unit_list
        return len(set(dose_unit_lists))
    
    @staticmethod
    def merge_two_med_dict(med_dict1, med_dict2):
        # merge two med_dict
        med_dict = med_dict1.copy()
        for medicine, dose_unit_list in med_dict2.items():
            if medicine not in med_dict.keys():
                med_dict[medicine] = dose_unit_list
            else:
                med_dict[medicine] += dose_unit_list
                med_dict[medicine] = list(set(med_dict[medicine]))
        return med_dict
    

if __name__ == "__main__":
    base_dir = dirname(dirname(realpath(__file__)))
    file_paths = [os.path.join(base_dir, "formatted_data\\synthea500_icd10_test\\concept.medication.csv")]
    statistic_medication = StatisticMedication(file_paths)
    statistic_medication()
