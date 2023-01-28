import pandas as pd
import os
from os.path import join, split


class MIMIC_CSV_to_Parquet_Converter:
    """Converts all csv files in a directory to parquet files"""
    def __init__(self, cfg, test=False):
        self.raw_data_dir = cfg.paths.raw_data_dir
        self.working_data_dir = cfg.paths.working_data_dir
        self.save_name = cfg.paths.save_name
        if self.save_name is None:
            self.save_name = split(self.raw_data_dir)[1]
        self.dest_dir = join(self.working_data_dir, 'interim', self.save_name)
        if not os.path.exists(self.dest_dir):
            os.makedirs(self.dest_dir)
        if test:
            self.nrows = 300

    def __call__(self):
        """
            test: load only 1000 rows
            file_name: path of the csv file, with ending
            dest_dir: directory to store the parquet files, without ending
        """
        files = get_csv_files_from_data_dir(self.raw_data_dir)
        for file in files:
            self.convert_csv_to_parquet(file)

    def convert_csv_to_parquet(self, file):
        src_file_path = join(self.raw_data_dir, file)
        columns = pandas_get_columns(src_file_path)
        dtype_dic = {column:"Int64" for column in columns if column.endswith('ID') and column!='FLUID'}
        dest_file_path = self.get_dest_path_for_parquet(src_file_path)
        pd.read_csv(src_file_path, nrows=self.nrows, dtype=dtype_dic, parse_dates=True, compression='gzip').to_parquet(dest_file_path, compression='gzip', index=False)

    def get_dest_path_for_parquet(self, file_path):
        file_name = split(file_path)[1].replace('.csv.gz', '.parquet.gz')
        return join(self.dest_dir, file_name)

def get_csv_files_from_data_dir(data_dir, exclude_files=[]):   
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv.gz') and not f in exclude_files]
    return files

def pandas_get_columns(path):
    if path.endswith('.gz'):
        compression = 'gzip'
    return pd.read_csv(path, nrows=1, compression=compression).columns
