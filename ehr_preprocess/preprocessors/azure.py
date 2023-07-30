import pandas as pd
import os
from azureml.core import Dataset
from tqdm import tqdm
import hashlib


class AzurePreprocessor():
    # load data in dask
    def __init__(self, cfg, logger, datastore) -> None:
        self.cfg = cfg
        self.logger = logger
        self.datastore =  datastore
        self.test = cfg.test
        self.logger.info(f"test {self.test}")
        self.removed_concepts = {k:0 for k in self.cfg.concepts.keys()} # count concepts that are removed
        self.initial_patients = set()
        self.formatted_patients = set()

    def __call__(self):
        self.patients_info()
        self.format_concepts()

    def format_concepts(self):
        """Loop over all top-level concepts (diagnosis, medication, procedures, etc.) and call processing"""
        admissions = self.get_admissions() # to assign admission_id
        for concept_type, concept_config in tqdm(self.cfg.concepts.items(), desc="Concepts"):
            if concept_type not in ['diagnosis', 'medication']:
                raise ValueError(f'{concept_type} not implemented yet')
            self.logger.info(f"INFO: Preprocess {concept_type}")
            first = True
            for chunk in tqdm(self.load_chunks(concept_config), desc='Chunks'):
                # process each chunk here.
                chunk_processed = self.concepts_process_pipeline(chunk, admissions, concept_type, concept_config)
                if first:
                    self.save(chunk_processed, concept_config, f'concept.{concept_type}', mode='w')
                    first = False
                else:
                    self.save(chunk_processed, concept_config, f'concept.{concept_type}', mode='a')
    
    def concepts_process_pipeline(self, concepts, admissions, concept_type, cfg):
        """Process concepts"""
        formatter = getattr(self, f"format_{concept_type}")
        concepts = formatter(concepts)
        self.initial_patients = self.initial_patients | set(concepts.PID.unique())
        self.logger.info(f"{len(self.initial_patients)} before cleaning")
        self.logger.info(f"{len(concepts)} concepts")
        concepts = concepts.dropna()
        self.logger.info(f"{len(concepts)} concepts after removing nans")
        concepts = concepts.drop_duplicates()
        self.logger.info(f"{len(concepts)} concepts after dropping duplicates nans")
        self.formatted_patients = self.formatted_patients | set(concepts.PID.unique())
        self.logger.info(f"{len(self.formatted_patients)} after cleaning")
        self.logger.info("Add admission id")
        concepts = self.add_admission_id(concepts, admissions)
        return concepts

    @staticmethod
    def format_diagnosis(diag):
        # Search code in diagnoses
        diag['code'] = diag['Diagnose'].str.extract(r'\((D.*?)\)', expand=False)
        diag['code'] = diag['code'].fillna(diag['Diagnose'])
        diag['CONCEPT'] = diag.Diagnosekode.fillna(diag.code)
        diag = diag.drop(['code', 'Diagnose', 'Diagnosekode'], axis=1)
        diag = diag.rename(columns={'CPR_hash':'PID', 'Noteret_dato':'TIMESTAMP'})
        return diag
    @staticmethod
    def format_medication(med):
        med.loc[:, 'CONCEPT'] = med.ATC.fillna('Ordineret_lægemiddel')
        med.loc[:, 'TIMESTAMP'] = med.Administrationstidspunkt.fillna("Bestillingsdato")
        med = med.rename(columns={'CPR_hash':'PID'})
        med = med[['PID','CONCEPT','TIMESTAMP']]
        med['CONCEPT'] = med['CONCEPT'].map(lambda x: 'M'+x)
        return med

    def patients_info(self):
        """Load patients info and rename columns"""
        self.logger.info("Load patients info")
        df = self.load_pandas(self.cfg.patients_info)
        if self.test:
            df = df.sample(10000)
        df = self.select_columns(df, self.cfg.patients_info)
        # Convert info dict to dataframe
        self.save(df, self.cfg.patients_info, 'patients_info')

    def add_admission_id(self, concept_df, adm_df):
        """
        Add unique admission IDs to records. For records within admission times,
        keep existing IDs. For others, generate IDs based on PID and timestamp.
        """
        # Filter records within and outside of admission times
        concept_df["EVENT_ID"] = range(len(concept_df))
        in_adm = self.filter_records_within_admission(concept_df, adm_df)

        in_adm_event_ids = in_adm.EVENT_ID.unique()
        out_adm = concept_df.loc[~concept_df.EVENT_ID.isin(in_adm_event_ids)]
        
        # Check that no event is in both in_adm and out_adm
        assert not any(out_adm.EVENT_ID.isin(in_adm_event_ids))
        # Assign unique admission IDs to records outside of admission times
        out_adm = self.assign_admission_id(out_adm)
        # Combine dataframes
        result_df = self.combine_dataframes(out_adm, in_adm)
        # Reset index to make PID a column again
        result_df = result_df.drop(columns=['EVENT_ID'])

        return result_df.reset_index(drop=True)


    @staticmethod
    def filter_records_within_admission(concept_df, adm_df):
        """
        Filter the records that fall within the admission time range and assign to closest admission.
        """
        # Reset index and sort values before the merge
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
        in_admission = (merged_df['TIMESTAMP']<=merged_df['TIMESTAMP_END']) & (merged_df['TIMESTAMP']>=merged_df['TIMESTAMP_START'])
        return merged_df[in_admission]
    @staticmethod
    def assign_admission_id(df):
        """
        Assign unique admission IDs to records based on PID and time difference.
        """
        df_sorted = df.sort_values(['TIMESTAMP'])
        df_sorted['TimeDiff'] = df_sorted.groupby('PID')['TIMESTAMP'].diff()
        df_sorted['NewID'] = df_sorted['TimeDiff'].apply(lambda x: 0 if pd.isnull(x) else int(x.total_seconds() > 24*60*60))

        df_sorted['ID'] = df_sorted.groupby('PID')['NewID'].cumsum().astype(int)
        df_sorted['ID'] = (df_sorted.PID.astype(str) + '_' + df_sorted['ID'].astype(str)).apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
        df_final = df_sorted.drop(columns=['TimeDiff', 'NewID']).rename(columns={'ID':'ADMISSION_ID'})
        return df_final
    @staticmethod
    def combine_dataframes(df1, df2):
        """
        Combine two dataframes, removing unnecessary columns from the one within admissions.
        """
        df2 = df2.drop(columns=['TIMESTAMP_START', 'TIMESTAMP_END'])
        return pd.concat([df1, df2])

    def get_admissions(self):
        """Load admission dataframe and create an ADMISSION_ID column"""
        self.logger.info("Load admissions")
        df = self.load_pandas(self.cfg.admissions)
        df = self.select_columns(df, self.cfg.admissions)
        df['ADMISSION_ID'] = self.assign_hash(df)
        return df
    
    @staticmethod
    def assign_hash(df):
        return df.apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest(), axis=1)

    def change_dtype(self, df, cfg):
        """Change column dtype"""
        if 'dtypes' in cfg:
            for col, dtype in cfg.dtypes.items():
                df[col] = df[col].astype(dtype)
        return df

    def select_columns(self, df, cfg):
        """Select and Rename columns"""
        columns = df.columns.tolist()
        selected_columns = [columns[i] for i in cfg.usecols]
        df = df[selected_columns]
        df = df.rename(columns={old: new for old, new in zip(selected_columns, cfg.names)})
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
        if 'start_chunk' in cfg:
            i = cfg.start_chunk
        else:
            i = 0
        while True:
            self.logger.info(f"chunk {i}")
            chunk = ds.skip(i * cfg.chunksize)
            chunk = chunk.take(cfg.chunksize)
            if pandas:
                df = chunk.to_pandas_dataframe()
            else:
                df = chunk.to_dask_dataframe()
            if len(df.index) == 0:
                self.logger.info("empty")
                break
            i += 1
            yield df
            
    def get_dataset(self, cfg: dict):
        ds = Dataset.Tabular.from_parquet_files(path=(self.datastore, cfg.filename))
        if 'keep_cols' in cfg:
            ds = ds.keep_columns(columns=cfg.keep_cols)
        if self.test:
            ds = ds.take(10000)
        return ds
    
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
            df.to_csv(path, index=False, mode=mode)
        else:
            raise ValueError(f"Filetype {file_type} not implemented yet")

    
