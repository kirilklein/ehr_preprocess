import pandas as pd
import os
from azureml.core import Dataset
from tqdm import tqdm
from os.path import join
from datetime import timedelta
from azure_run import datastore
import formatters
from utils import assign_hash
from load import load_pandas, load_chunks

class AzurePreprocessor():
    # load data in dask
    def __init__(self, cfg, logger) -> None:
        self.cfg = cfg
        self.logger = logger
        self.test = cfg.test
        self.logger.info(f"test {self.test}")
        self.initial_patients = set()
        self.formatted_patients = set()
        self.adm_file = None

    def __call__(self):
        self.patients_info()
        self.format_concepts()

    def format_concepts(self):
        """Loop over all top-level concepts (diagnosis, medication, procedures, etc.) and call processing"""
        self.get_admissions() # to assign admission_id

        # Getting SP concepts
        if 'SP_concepts' in self.cfg:
            self.logger.info("Load SP concepts")
            for concept_type, concept_config in tqdm(self.cfg.SP_concepts.types.items(), desc="Concepts"):
                if concept_type not in ['diagnosis', 'medication', 'labtest', 'procedure']:
                    raise ValueError(f'{concept_type} not implemented yet')
                self.logger.info(f"INFO: Preprocess {concept_type}")
                first = True
                concept_config.data_path = join(self.cfg.SP_concepts.dump_path, concept_config.filename)
                concept_config.data_store = self.cfg.SP_concepts.data_store
                if isinstance(concept_config.filename, list):
                    for file_name in concept_config.filename:
                        concept_config.filename = file_name
                        self.iterate_through_file(concept_type, concept_config, first=first)
                        first=False
                else:
                    self.iterate_through_file(concept_type, concept_config)

        if 'register_concepts' in self.cfg:
            raise NotImplementedError("register_concepts not implemented yet")
        self.save(adm_file, self.cfg.admissions, 'admissions')

    def iterate_through_file(self, concept_type, concept_config, first=True):
        for chunk in tqdm(self.load_chunks(concept_config), desc='Chunks'):
            chunk_processed = self.concepts_process_pipeline(chunk, concept_type, concept_config)
            if first:
                self.save(chunk_processed, concept_config, f'concept.{concept_type}', mode='w')
                first = False
            else:
                self.save(chunk_processed, concept_config, f'concept.{concept_type}', mode='a')

    def concepts_process_pipeline(self, concepts, concept_type, cfg):
        """Process concepts"""
        formatter = getattr(formatters, f"format_{concept_type}")
        concepts = formatter(concepts, cfg)
        self.initial_patients = self.initial_patients | set(concepts.PID.unique())
        self.logger.info(f"{len(self.initial_patients)} before cleaning")
        self.logger.info(f"{len(concepts)} concepts")
        concepts = concepts.dropna()
        self.logger.info(f"{len(concepts)} concepts after removing nans")
        concepts = concepts.drop_duplicates()
        self.logger.info(f"{len(concepts)} concepts after dropping duplicates nans")
        filter_date = self.cfg.filtering.get('filter_date', False) if hasattr(self.cfg, 'filtering') and self.cfg.filtering else False
        if filter_date:
            concepts = self.filter_dates(concepts, filter_date)
        self.logger.info(f"{len(concepts)} concepts after filtering on date")
        self.formatted_patients = self.formatted_patients | set(concepts.PID.unique())
        self.logger.info(f"{len(self.formatted_patients)} after cleaning")
        self.logger.info("Add admission id")
        concepts = self.add_admission_id(concepts)
        return concepts
    
    def filter_dates(self, chunk, filter_date):
        chunk['TIMESTAMP'] = pd.to_datetime(chunk['TIMESTAMP'])
        filter_date_dt = pd.to_datetime(filter_date)
        filtered_chunk = chunk[chunk['TIMESTAMP'] < filter_date_dt]
        return filtered_chunk

    def patients_info(self):
        """Load patients info and rename columns"""
        self.logger.info("Load patients info")
        config = self.cfg.patients_info
        self.cfg.patients_info.data_path = join(config.dump_path, config.filename)
        df = self.load_pandas(self.cfg.patients_info)
        if self.test:
            df = df.sample(10000)
        df = self.select_columns(df, self.cfg.patients_info)
        # Convert info dict to dataframe
        self.save(df, self.cfg.patients_info, 'patients_info')

    def add_admission_id(self, concept_df):
        """
        Add unique admission IDs to records. For records within admission times,
        keep existing IDs. For others, generate IDs based on PID and timestamp.
        """
        # Filter records within and outside of admission times
        in_adm, out_adm = self.filter_records_with_exisiting_admission(concept_df, self.adm_file)
        # Assign unique admission IDs to records outside of admission times
        out_adm, self.adm_file = self.assign_admission_id(out_adm, self.adm_file)
        # Combine dataframes
        result_df = pd.concat([in_adm, out_adm])
        result_df = result_df.drop(columns=['TIMESTAMP_START', 'TIMESTAMP_END', 'TYPE'])
        return result_df.reset_index(drop=True)

    @staticmethod
    def filter_records_with_exisiting_admission(concept_df, adm_df):
        """
        Filter the records that fall within the admission time range and assign to closest admission.
        """
        # Reset index and sort values before the merge
        concept_df['TIMESTAMP'] = pd.to_datetime(concept_df['TIMESTAMP'])
        concept_df = concept_df.reset_index().sort_values("TIMESTAMP")
        adm_df = adm_df.reset_index().sort_values("TIMESTAMP_START")

        # Merge on PID with outer join to get all combinations
        merged_df = pd.merge_asof(
            concept_df, 
            adm_df, 
            left_on="TIMESTAMP",
            right_on="TIMESTAMP_START",
            by="PID", 
            direction="nearest"
        ).drop(columns=["index_x", "index_y"])
        
        # Filter to keep only the rows where TIMESTAMP is within the admission time range
        in_admission = (
            (merged_df['TIMESTAMP'] <= merged_df['TIMESTAMP_END'] + pd.Timedelta(days=1)) &
            (merged_df['TIMESTAMP'] >= merged_df['TIMESTAMP_START'] - pd.Timedelta(days=1)) &
            (merged_df['TYPE'] == 'IN')
        )
        out_admission = (
            (merged_df['TIMESTAMP'] <= merged_df['TIMESTAMP_END'] + pd.Timedelta(days=2)) &
            (merged_df['TIMESTAMP'] >= merged_df['TIMESTAMP_START'] - pd.Timedelta(days=2)) &
            (merged_df['TYPE'] == 'OUT')
        )
        existing_admissions = in_admission | out_admission
        return merged_df[existing_admissions], merged_df[~existing_admissions]

    @staticmethod
    def assign_admission_id(df, adm_file):
        """
        Assign unique admission IDs to records outside of hospital admissions based on PID and time difference.
        Here all records 24 hours of each other are considered to be in the same admission.
        """
        df_sorted = df.sort_values(['PID', 'TIMESTAMP'])

        # Calculate time differences within each PID group
        df_sorted['TIMESTAMP_DIFF'] = df_sorted.groupby('PID')['TIMESTAMP'].diff().fillna(pd.Timedelta(seconds=0))
        
        # Identify new admissions based on the time difference
        df_sorted['NEW_ADMISSION'] = (df_sorted['TIMESTAMP_DIFF'] > pd.Timedelta(days=2)).cumsum()

        # Generate unique admission IDs
        df_sorted['ADMISSION_ID'] = df_sorted.apply(
            lambda row: hashlib.sha256((str(row['PID']) + '_' + str(row['NEW_ADMISSION'])).encode()).hexdigest(), axis=1
        )

        new_admissions = df_sorted.groupby(['PID', 'NEW_ADMISSION']).agg(
            TIMESTAMP_START=('TIMESTAMP', 'min'),
            TIMESTAMP_END=('TIMESTAMP', 'max')
        ).reset_index()
        new_admissions['TYPE'] = 'OUT'

        new_admissions['ADMISSION_ID'] = new_admissions.apply(
            lambda row: hashlib.sha256((str(row['PID']) + '_' + str(row['NEW_ADMISSION'])).encode()).hexdigest(), axis=1
        )

        adm_file = pd.concat([adm_file, new_admissions[['PID', 'ADMISSION_ID', 'TIMESTAMP_START', 'TIMESTAMP_END', 'TYPE']]], ignore_index=True)

        return df_sorted.drop(columns=['TIMESTAMP_DIFF', 'NEW_ADMISSION']), adm_file

    def get_admissions(self):
        """Load admission dataframe and create an ADMISSION_ID column. Then combined all admission within 24 hours."""
        self.logger.info("Load admissions")
        config = self.cfg.admissions
        self.cfg.admissions.data_path = join(config.dump_path, config.filename)
        adm = self.load_pandas(self.cfg.admissions)
        adm['Flyt_ind'] = pd.to_datetime(adm['Flyt_ind'])
        adm['Flyt_ud'] = pd.to_datetime(adm['Flyt_ud'])
        
        adm.sort_values(by=['CPR_hash', 'Flyt_ind'], inplace=True)
        merged_admissions = []
        current_row = None
        for _, row in adm.iterrows():
            if current_row is None:
                current_row = row
                continue

            # Check for overlap or if next admission is within 24 hours after the current admission's discharge
            if row['Flyt_ind'] <= current_row['Flyt_ud'] + timedelta(hours=24) and row['CPR_hash'] == current_row['CPR_hash']:
                # Extend the current admission's discharge time if the next admission's discharge time is later
                current_row['Flyt_ud'] = max(current_row['Flyt_ud'], row['Flyt_ud'])
            else:
                # No overlap within 24 hours, add the current admission to merged_admissions and start a new current admission
                merged_admissions.append(current_row.to_dict())
                current_row = row
        merged_admissions.append(current_row.to_dict())

        events = []
        for admission in merged_admissions:
            events.append({'PID': admission['CPR_hash'], 
                           'TIMESTAMP_START': admission['Flyt_ind'], 
                           'TIMESTAMP_END': admission['Flyt_ud'],
                           'TYPE': 'IN'})
        final_df = pd.DataFrame(events)
        final_df['ADMISSION_ID'] = self.assign_hash(final_df)
        self.adm_file = final_df

    def select_columns(self, df, cfg):
        """Select and Rename columns"""
        columns = df.columns.tolist()
        selected_columns = [columns[i] for i in cfg.usecols]
        df = df[selected_columns]
        df = df.rename(columns={old: new for old, new in zip(selected_columns, cfg.names)})
        return df
            
    def get_researcher_data(path, 
                            n_take: int | None = None,
                            separator: str = ",",
                            encoding: str = "utf8"):
        base_path = f'https://forskerpln0ybkrdls01.blob.core.windows.net/researcher-data/'
        if ".parquet" in path:
            ds = Dataset.Tabular.from_parquet_files(path=base_path + path)
        elif ".csv" in path or ".asc" in path:
            ds = Dataset.Tabular.from_delimited_files(path=base_path + path, separator=separator, encoding=encoding)
        else:
            print("invalid filetype")
            return None
        
        if n_take:
            df = ds.take(n_take).to_pandas_dataframe()
        else:
            df = ds.to_pandas_dataframe()
        return df

    def load_pandas(self, cfg: dict):
        ds = self.get_dataset(cfg)
        df = ds.to_pandas_dataframe()
        return df

    def load_dask(self, cfg: dict):
        ds = self.get_dataset(cfg)
        df = ds.to_dask_dataframe()
        return df

    def load_chunks(self, cfg: dict, pandas=True):
        """Generate chunks of the dataset and convert to pandas/dask df"""
        ds = self.get_dataset(cfg)
        i = cfg.start_chunk if 'start_chunk' in cfg else 0
        while True:
            self.logger.info(f"chunk {i}")
            chunk = ds.skip(i * cfg.chunksize)
            chunk = chunk.take(cfg.chunksize)
            df = chunk.to_pandas_dataframe() if pandas else chunk.to_dask_dataframe()
            if len(df.index) == 0:
                self.logger.info("empty")
                break
            i += 1
            yield df

    def get_dataset(self, cfg: dict):
        file_path = cfg.data_path
        ds_store = datastore(cfg.data_store)

        if 'parquet' in file_path:
            ds = Dataset.Tabular.from_parquet_files(path=(ds_store,file_path))
        elif ".csv" in file_path or ".asc" in file_path:
            try:
                ds = Dataset.Tabular.from_delimited_files(path=(ds_store,file_path), separator=';', encoding='utf-8')
            except UnicodeDecodeError:
                ds = Dataset.Tabular.from_delimited_files(path=(ds_store,file_path), separator=';', encoding='iso-8859-1')
        if 'keep_cols' in cfg:
            ds = ds.keep_columns(columns=cfg.keep_cols)
        if self.test:
            ds = ds.take(10000)
        return ds
    
    def save(self, df, cfg, filename, mode='w'):
        self.logger.info(f"Save {filename}")
        out = self.cfg.paths.output_dir
        file_type = cfg.file_type if 'file_type' in cfg else self.cfg.file_type
        try:
            os.makedirs(out, exist_ok=True)
            if file_type == 'parquet':
                path = os.path.join(out, f'{filename}.parquet')
                df.to_parquet(path)
            elif file_type == 'csv':
                path = os.path.join(out, f'{filename}.csv')
                df.to_csv(path, index=True, mode=mode, header=(mode == 'w'))
            else:
                raise ValueError(f"Filetype {file_type} not implemented yet")
        except (OSError, IOError) as e:
            self.logger.error(f"Failed to save {filename}: {str(e)}")
            raise
