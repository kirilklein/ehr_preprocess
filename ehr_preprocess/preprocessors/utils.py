import pandas as pd
import os
from os.path import join, split
from tqdm import tqdm

class MIMIC_CSV_to_Parquet_Converter:
    """Converts all csv files in a directory to parquet files"""
    def __init__(self, cfg, test=False, overwrite=False):
        self.raw_data_dir = cfg.paths.raw_data_dir
        self.working_data_dir = cfg.paths.working_data_dir
        self.save_name = cfg.paths.save_name
        self.overwrite = overwrite
        if self.save_name is None:
            self.save_name = split(self.raw_data_dir)[1]
        self.dest_dir = join(self.working_data_dir, 'interim', self.save_name)
        if not os.path.exists(self.dest_dir):
            os.makedirs(self.dest_dir)
        self.nrows = None
        if test:
            self.nrows = 300
            self.dest_dir = join(split(self.dest_dir)[0], 'test')
            os.makedirs(self.dest_dir)

    def __call__(self):
        """
            test: load only 1000 rows
            file_name: path of the csv file, with ending
            dest_dir: directory to store the parquet files, without ending
        """
        files = get_csv_files_from_data_dir(self.raw_data_dir)
        print(f"Convert {len(files)} files")
        for file in tqdm(files):
            print('File: ', file)
            self.convert_csv_to_parquet(file)

    def convert_csv_to_parquet(self, file):
        src_file_path = join(self.raw_data_dir, file)
        columns = pandas_get_columns(src_file_path)
        dtype_dic = {column:"Int64" for column in columns if column.endswith('ID') and column!='FLUID'}
        self.load_columns = self.get_load_columns(file, columns)
        if 'CHARTEVENTS' in file:
            dtype_dic['VALUE'] = str
        
        dest_file_path = self.get_dest_path_for_parquet(src_file_path)
        file_size = os.path.getsize(join(self.raw_data_dir, file))/1e9
        if file_size > 1:
            print('File too big: ', dest_file_path)
            print('Process in chunks')
            chunk = True
            chunksize = 10e6
        else:
            chunk = False
            chunksize = None
        if os.path.isfile(dest_file_path):
            print('File already exists: ', dest_file_path)
            if self.overwrite:
                print('Overwrite')
                self.conversion(src_file_path, dest_file_path, dtype_dic, chunk, chunksize=chunksize)
        else:
            print('Convert: ', split(src_file_path)[1], ' to ', split(dest_file_path)[1])
            self.conversion(src_file_path, dest_file_path, dtype_dic, chunk, chunksize=chunksize)
    
    def get_load_columns(file, columns):
        if file in drop_columns_dic.keys():
            drop_columns = drop_columns_dic[file]
        else:
            drop_columns = []
        load_columns = [column for column in columns if column not in drop_columns]
        return load_columns

    def conversion(self, src_file_path, dest_file_path, dtype_dic, chunk=False, chunksize=None, na_values=['No',]):
        if not chunk:
            pd.read_csv(src_file_path, nrows=self.nrows, dtype=dtype_dic, parse_dates=True, na_values=na_values, compression='gzip', uscols=self.load_columns).to_parquet(dest_file_path, compression='gzip', index=False)
        else:
            for i, df_chunk in enumerate(pd.read_csv(src_file_path, chunksize=chunksize, compression='gzip', dtype=dtype_dic, parse_dates=True, na_values=na_values, usecols=self.load_columns)):
                print('chunk: ', i)
                if not os.path.isfile(dest_file_path.replace('.parquet', f'_{i}.parquet')):
                    try:
                        df_chunk.to_parquet(dest_file_path.replace('.parquet', f'_{i}.parquet'), compression='gzip', index=False)
                    except:
                        continue

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

drop_columns_dic = {
    'CHARTEVENTS.csv.gz': ['RESULTSTATUS', 'STOPPED','WARNING', 'STORETIME', 'CGID', 'VALUENUM', 'ROW_ID'],
    'DATETIMEEVENTS.csv.gz': ['ROW_ID', 'VALUEUOM', 'WARNING', 'RESULTSTATUS', 'STOPPED',
}