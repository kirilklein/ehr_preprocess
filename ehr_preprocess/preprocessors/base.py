import json
import os
from os.path import dirname, join, realpath, split

base_dir = dirname(dirname(dirname(realpath(__file__))))

class BasePreprocessor():
    def __init__(self, cfg, test=False) -> None:
        self.test = test
        self.cfg = cfg
        self.raw_data_path = cfg.paths.raw_data_path
        self.prepends = cfg.prepends
        data_folder_name = split(self.raw_data_path)[-1]

        if not cfg.paths.working_data_path is None:
            working_data_path = cfg.paths.working_data_path
        else:
            working_data_path = join(base_dir, 'data')

        self.interim_data_path = join(
            working_data_path, 'interim', data_folder_name)
        if not os.path.exists(self.interim_data_path):
            os.makedirs(self.interim_data_path)
        self.formatted_data_path = os.getcwd()
        if not os.path.exists(self.formatted_data_path):
            os.makedirs(self.formatted_data_path)
        if test:
            self.nrows = 10000
        else:
            self.nrows = None
        if os.path.exists(join(self.formatted_data_path, 'metadata.json')):
            with open(join(self.formatted_data_path, 'metadata.json'), 'r') as fp:
                self.metadata_dic = json.load(fp)
        else:
            self.metadata_dic = {}

    def save_metadata(self):
        with open(join(self.formatted_data_path, 'metadata.json'), 'w') as fp:
            json.dump(self.metadata_dic, fp)

    def update_metadata(self, concept_name, coding_system, src_files_ls):
        if concept_name!='patients_info':
            file = f'concept.{concept_name}.parquet'
            prepend = self.prepends[concept_name]
        else:
            file = 'patients_info.parquet'
            prepend = None
        concept_dic = {
            'Coding_System': coding_system, 'Prepend': prepend, 'Source': src_files_ls
        }
        if file not in self.metadata_dic:
            self.metadata_dic[file] = concept_dic
    @staticmethod
    def drop_missing_timestamps(df):
        if 'TIMESTAMP' in df.columns:
            df['TIMESTAMP'] = df['TIMESTAMP'].dropna()
        return df


    