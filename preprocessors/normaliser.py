import pandas as pd
from os.path import join
from pathlib import Path
import os
from tqdm import tqdm
from azureml.core import Dataset
from preprocessors.azure import AzurePreprocessor
import torch
import numpy as np

class Normaliser():
    # load data in dask
    def __init__(self, cfg, logger, datastore) -> None:
        self.cfg = cfg
        self.logger = logger
        self.datastore =  datastore
        self.test = cfg.test
        self.logger.info(f"test {self.test}")
        self.firstRound = True
        self.normalisation_type = cfg.data['norm_type']
        self.azure_processor = AzurePreprocessor(cfg, logger, datastore)

        # Load distribution data
        dist_path = join(self.cfg.paths.dist_path)
        dist_dataset = Dataset.File.from_files(path=(datastore, dist_path))
        mount_context = dist_dataset.mount()
        mount_context.start()
        mount_point = mount_context.mount_point
        dist = torch.load(join(mount_point, 'lab_val_dict.pt'))
        self.vocab = torch.load(join(mount_point, 'vocabulary.pt'))        

        if self.normalisation_type == 'Min_max':
            # Gets the min max values for all concepts
            min_max_dict = {
                concept: (
                    np.percentile(dist[concept], 0.01 * 100) if len(dist[concept]) > 1 else dist[concept][0],
                    np.percentile(dist[concept], 0.99 * 100) if len(dist[concept]) > 1 else dist[concept][0]
                ) 
                for concept in dist 
                if dist[concept]
            }
            self.min_max_vals = min_max_dict

        elif self.normalisation_type == 'Categorise':
            self.quantiles = {}
            for concept in dist:
                sorted_dist = sorted(dist[concept])
                if len(sorted_dist) > 0:
                    q1 = np.percentile(sorted_dist, 25)
                    q2 = np.percentile(sorted_dist, 50)
                    q3 = np.percentile(sorted_dist, 75)
                    self.quantiles[concept] = (q1, q2, q3)
                else:
                    self.quantiles[concept] = (0, 0, 0)

        elif self.normalisation_type == 'Quantiles':
            self.n_quantiles = cfg.data['n_quantiles']
            self.quantiles = {}
            for concept in dist:
                sorted_dist = sorted(dist[concept])
                if len(sorted_dist) > 0:
                    # Calculate percentiles for 10 quantiles (10, 20, ..., 100)
                    quantiles = [np.percentile(sorted_dist, i) for i in np.linspace(100/self.n_quantiles, 100, self.n_quantiles)]
                    self.quantiles[concept] = quantiles
                else:
                    self.quantiles[concept] = [0] * self.n_quantiles
        else:
            raise ValueError('Invalid type of normalisation')

    def __call__(self):
        cfg = self.cfg
        save_name = cfg.paths.save_name
        if not Path(join(cfg.paths.output_dir, save_name)).exists():
            counter = 0
            # Iterate over chunks of the CSV file
            for chunk in tqdm(self.azure_processor.load_chunks(cfg), desc='Chunks'):
                self.logger.info(f'Loaded {cfg.data.chunksize*counter}')
                chunk_processed = self.process_chunk(chunk)
                if counter == 0:
                    self.azure_processor.save(chunk_processed, cfg.paths,  f'concept.{save_name}', mode='w')
                else:
                    self.azure_processor.save(chunk_processed, cfg.paths, f'concept.{save_name}', mode='a')
                
                counter += 1

    def process_chunk(self, chunk):
        chunk['RESULT'] = chunk.apply(self.normalise, axis=1)
        return chunk
        
    def normalise(self, row):
        if not row['CONCEPT'] in self.vocab:
            return row['RESULT']
        
        concept = self.vocab[row['CONCEPT']]
        value = row['RESULT']        
        # Returns value if it is not numerical
        if not pd.notnull(pd.to_numeric(value, errors='coerce')):
            return value
        else: 
            value = pd.to_numeric(value)

        # Normalises numerical values
        if self.normalisation_type == 'Min_max':
            return self.min_max_normalise(concept, value)
        elif self.normalisation_type == 'Quantiles':
            return self.quantile(concept, value)
        else:
            Warning(f"Normalisation type {self.normalisation_type} not implemented")
        
    def min_max_normalise(self, concept, value):
        if concept in self.min_max_vals:
            (min_val, max_val) = self.min_max_vals[concept]
            if max_val != min_val: 
                normalised_value = (value - min_val) / (max_val - min_val)
                return max(0, min(1, normalised_value))
            else:
                return 0
        else:
            return 0
        
    def quantile(self, concept, value):
        if concept not in self.quantiles:
            return 'N/A'
        else:
            quantile_values = self.quantiles[concept]
            # Ensure there are exactly 12 quantiles
            if len(quantile_values) != self.n_quantiles:
                raise ValueError(f"Expected {self.n_quantiles} quantiles for concept '{concept}'")            
            for i, q in enumerate(quantile_values, start=1):
                if value <= q:
                    return 'Q{}'.format(i)
            return 'Q{}'.format(self.n_quantiles)

class DataHandler():
    def __init__(self, cfg, logger, datastore) -> None:
        self.cfg = cfg
        self.logger = logger
        self.datastore =  datastore
        self.test = cfg.test
        self.logger.info(f"Datahandler initialised")
        self.firstRound = True 

    def updateFirstRound(self):
        self.firstRound = False 

    def save(self, df, cfg, filename, mode='w'):
        self.logger.info(f"Save {filename}")
        out = self.cfg.paths.output_dir
        if 'file_type' in cfg:
            file_type = cfg.file_type
        else:
            file_type = self.cfg.file_type
        if not os.path.exists(out):
            os.makedirs(out)
        if file_type == 'parquet':
            path = os.path.join(out, f'{filename}.parquet')
            df.to_parquet(path)
        elif file_type == 'csv':
            path = os.path.join(out, f'{filename}.csv')
            if mode == 'w':
                df.to_csv(path, index=True, mode=mode)
            else: 
                df.to_csv(path, index=True, mode=mode, header=False)
        else:
            raise ValueError(f"Filetype {file_type} not implemented yet")

    def load_chunks(self, cfg: dict, pandas=True):
        """Generate chunks of the dataset and convert to pandas/dask df"""
        ds = self.get_dataset()
        if 'start_chunk' in cfg:
            i = cfg.start_chunk
        else:
            i = 0
        while True:
            self.logger.info(f"chunk {i}")
            chunk = ds.skip(i * cfg.data.chunksize)
            chunk = chunk.take(cfg.data.chunksize)
            if pandas:
                df = chunk.to_pandas_dataframe()
            else:
                df = chunk.to_dask_dataframe()
            if len(df.index) == 0:
                self.logger.info("empty")
                break
            i += 1
            yield df
            
    def get_dataset(self):
        file_path = join(self.cfg.paths.file_path, self.cfg.paths.file_name)

        if self.firstRound:
            ds = Dataset.Tabular.from_delimited_files(path=(self.datastore, file_path))
            if self.test:
                ds = ds.take(10000)
            return ds
        else:
            ds = Dataset.Tabular.from_delimited_files(path=(self.cfg.paths.file_path, self.cfg.paths.file_name))
            return ds




