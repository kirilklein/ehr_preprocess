import pandas as pd
import os
from azureml.core import Dataset
from tqdm import tqdm
import hashlib
from azureml.core import Workspace, Dataset, Datastore
from os.path import join


class AzurePreprocessor():
    # load data in dask
    def __init__(self, cfg, logger, datastore, dump_path) -> None:
        self.cfg = cfg
        self.logger = logger
        self.datastore =  datastore
        self.dump_path = dump_path if dump_path is not None else None
        self.test = cfg.test
        self.logger.info(f"test {self.test}")
        self.removed_concepts = {k:0 for k in self.cfg.concepts.keys()} # count concepts that are removed
        self.initial_patients = set()
        self.formatted_patients = set()
        self.out_adm_file = None

    def __call__(self):
        self.patients_info()
        self.format_concepts()

    def filter_dates(self):
        self.logger.info("Filter dates")
        for concept_type, concept_config in tqdm(self.cfg.concepts.items(), desc="Concepts"):
            if concept_type not in ['diagnosis', 'medication', 'labtest', 'procedure']:
                raise ValueError(f'{concept_type} not implemented yet')
            self.logger.info(f"INFO: Filter {concept_type}")            
            df = self.load_pandas(concept_config)
            if self.test:
                df = df.sample(10000)
            df = self.select_columns(df, concept_config)
            df = self.change_dtype(df, concept_config)
            df = self.filter_dates_pipeline(df, concept_config)
            self.save(df, concept_config, f'concept.{concept_type}')

    def iterate_through_file(self, admissions, concept_type, concept_config, first=True):
        for chunk in tqdm(self.load_chunks(concept_config), desc='Chunks'):
            # process each chunk here.
            chunk_processed = self.concepts_process_pipeline(chunk, admissions, concept_type, concept_config)
            if first:
                self.save(chunk_processed, concept_config, f'concept.{concept_type}', mode='w')
                first = False
            else:
                self.save(chunk_processed, concept_config, f'concept.{concept_type}', mode='a')

    def format_concepts(self):
        """Loop over all top-level concepts (diagnosis, medication, procedures, etc.) and call processing"""
        admissions = self.get_admissions() # to assign admission_id
        for concept_type, concept_config in tqdm(self.cfg.concepts.items(), desc="Concepts"):
            if concept_type not in ['diagnosis', 'medication', 'labtest', 'procedure']:
                raise ValueError(f'{concept_type} not implemented yet')
            self.logger.info(f"INFO: Preprocess {concept_type}")
            first = True

            if type(concept_config.filename) == list:
                for file_name in concept_config.filename:
                    concept_config.filename = file_name
                    self.iterate_through_file(admissions, concept_type, concept_config, first=first)
                    first=False
            else:
                self.iterate_through_file(admissions, concept_type, concept_config)
        combine_and_save_admissions = self.save_adm(admissions, self.out_adm_file)


    def concepts_process_pipeline(self, concepts, admissions, concept_type, cfg):
        """Process concepts"""
        formatter = getattr(self, f"format_{concept_type}")
        concepts = formatter(concepts, cfg.formatter_args)
        self.initial_patients = self.initial_patients | set(concepts.PID.unique())
        self.logger.info(f"{len(self.initial_patients)} before cleaning")
        self.logger.info(f"{len(concepts)} concepts")
        concepts = concepts.dropna()
        self.logger.info(f"{len(concepts)} concepts after removing nans")
        concepts = concepts.drop_duplicates()
        self.logger.info(f"{len(concepts)} concepts after dropping duplicates nans")
        filter_date = self.cfg.filtering.get('filter_date', False) if hasattr(self.cfg, 'filtering') and self.cfg.filtering else False
        if filter_date:
            concepts = self.filter_dates_pipeline(concepts, filter_date)
        self.logger.info(f"{len(concepts)} concepts after filtering on date")
        self.formatted_patients = self.formatted_patients | set(concepts.PID.unique())
        self.logger.info(f"{len(self.formatted_patients)} after cleaning")
        self.logger.info("Add admission id")
        concepts = self.add_admission_id(concepts, admissions)
        return concepts
    
    def filter_dates_pipeline(self, chunk, filter_date):
        chunk['TIMESTAMP'] = pd.to_datetime(chunk['TIMESTAMP'])
        filter_date_dt = pd.to_datetime(filter_date)
        filtered_chunk = chunk[chunk['TIMESTAMP'] < filter_date_dt]
        return filtered_chunk

    @staticmethod
    def format_diagnosis(diag, args):
        # Search code in diagnoses. If there is no diagnosis code, use the diagnosis extracted from the text
        diag['code'] = diag['Diagnose'].str.extract(r'\((D.*?)\)', expand=False)
        if 'fill_diags' in args and args['fill_diags']:
            diag['code'] = diag['code'].fillna(diag['Diagnose'])
        diag['CONCEPT'] = diag.Diagnosekode.fillna(diag.code)
        diag = diag.drop(['code', 'Diagnose', 'Diagnosekode'], axis=1)
        diag = diag.rename(columns={'CPR_hash':'PID', 'Noteret_dato':'TIMESTAMP'})
        return diag

    @staticmethod
    def format_procedure(proc, args):
        proc['CONCEPT'] = proc['ProcedureCode'].str.replace(' ', '')
        proc = proc.drop(['ProcedureCode'], axis=1)
        proc = proc.rename(columns={'CPR_hash':'PID', 'ServiceDatetime':'TIMESTAMP'})
        proc['CONCEPT'] = proc['CONCEPT'].map(lambda x: 'P'+x)
        return proc

    @staticmethod
    def format_labtest(labs, args):
        labs = labs.rename(columns={'CPR_hash':'PID', 'BestOrd':'CONCEPT', 'Bestillingsdato': 'TIMESTAMP', 'Resultatværdi':'RESULT'})
        labs['CONCEPT'] = labs['CONCEPT'].map(lambda x: 'LAB'+x)
        return labs
    
    @staticmethod
    def format_medication(med, args):
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
        in_adm, out_adm = self.filter_records_within_admission(concept_df, adm_df)
        
        # Assign unique admission IDs to records outside of admission times
        out_adm, self.out_adm_file = self.assign_admission_id(out_adm, self.out_adm_file)
        # Combine dataframes
        result_df = self.combine_dataframes(out_adm, in_adm)
        # Reset index to make PID a column again
        result_df = result_df.drop(columns=['EVENT_ID', 'ADMISSION', 'DISCHARGE'])

        return result_df.reset_index(drop=True)


    @staticmethod
    def filter_records_within_admission(concept_df, adm_df):
        """
        Filter the records that fall within the admission time range and assign to closest admission.
        """
        # Reset index and sort values before the merge
        concept_df = concept_df.reset_index().sort_values("TIMESTAMP")
        adm_df = adm_df.reset_index().sort_values("ADMISSION")

        # Merge on PID with outer join to get all combinations
        merged_df = pd.merge_asof(
            concept_df, 
            adm_df, 
            left_on="TIMESTAMP",
            right_on="ADMISSION",
            by="PID", 
            direction="nearest"
        ).drop(columns=["index_x", "index_y"])
        
        # Filter to keep only the rows where TIMESTAMP is within the admission time range
        in_admission = (merged_df['TIMESTAMP']<=merged_df['DISCHARGE']) & (merged_df['TIMESTAMP']>=merged_df['ADMISSION'])
        return merged_df[in_admission], merged_df[~in_admission]

    @staticmethod
    def assign_admission_id(df, out_adm_file):
        """
        Assign unique admission IDs to records outside of hospital admissions based on PID and time difference. 
        Here all records 24 hours of each other are considered to be in the same admission.
        """
        df_sorted = df.sort_values(['TIMESTAMP'])

        if out_adm_file is None:
            out_adm_file = pd.DataFrame(columns=['PID', 'ADMISSION_ID', 'TIMESTAMP_START', 'TIMESTAMP_END'])

        for index, row in df_sorted.iterrows():
            pid = row['PID']
            timestamp = row['TIMESTAMP']
            
            # Find matching admission periods within ±24 hours
            matching_adm = out_adm_file[
                (out_adm_file['PID'] == pid) &
                (out_adm_file['TIMESTAMP_START'] - pd.Timedelta(hours=24) <= timestamp) &
                (out_adm_file['TIMESTAMP_END'] + pd.Timedelta(hours=24) >= timestamp)
            ]
            
            if not matching_adm.empty:
                # Assign the corresponding ADMISSION_ID
                df_sorted.at[index, 'ADMISSION_ID'] = matching_adm.iloc[0]['ADMISSION_ID']
                
                # Update the TIMESTAMP_START and TIMESTAMP_END in out_adm_file
                if timestamp < matching_adm.iloc[0]['TIMESTAMP_START']:
                    out_adm_file.loc[matching_adm.index, 'TIMESTAMP_START'] = timestamp
                if timestamp > matching_adm.iloc[0]['TIMESTAMP_END']:
                    out_adm_file.loc[matching_adm.index, 'TIMESTAMP_END'] = timestamp
            else:
                # Create a new ADMISSION_ID
                new_admission_id = hashlib.sha256((str(pid) + '_' + str(timestamp)).encode()).hexdigest()
                df_sorted.at[index, 'ADMISSION_ID'] = new_admission_id
                
                # Add a new row to out_adm_file
                new_row = {
                    'PID': pid,
                    'ADMISSION_ID': new_admission_id,
                    'TIMESTAMP_START': timestamp,
                    'TIMESTAMP_END': timestamp + pd.Timedelta(days=7)
                }
                out_adm_file = out_adm_file.append(new_row, ignore_index=True)

        return df_sorted, out_adm_file

    @staticmethod
    def save_adm(in_adm_file, out_adm_file):
        """
        Save the admission files to the output directory.
        """
        in_adm_file = in_adm_file.rename(columns={'ADMISSION': 'TIMESTAMP_START', 'DISCHARGE': 'TIMESTAMP_END'})
        in_adm_file['TYPE'] = 'IN'
        out_adm_file['TYPE'] = 'OUT'
        merged_adm_file = pd.concat([in_adm_file, out_adm_file])
        self.save(merged_adm_file, self.cfg.admissions, 'admissions')
        return in_adm_file, out_adm_file

    @staticmethod
    def combine_dataframes(df1, df2):
        """
        Combine two dataframes, removing unnecessary columns from the one within admissions.
        """
        df2 = df2.drop(columns=['TIMESTAMP_START', 'TIMESTAMP_END'])
        return pd.concat([df1, df2])

    def get_admissions(self):
        """Load admission dataframe and create an ADMISSION_ID column. Then combined all admission within 24 hours."""
        self.logger.info("Load admissions")
        adm = self.load_pandas(self.cfg.admissions)
        adm['Flyt_ind'] = pd.to_datetime(adm['Flyt_ind'])
        adm['Flyt_ud'] = pd.to_datetime(adm['Flyt_ud'])
        
        adm.sort_values(by=['CPR_hash', 'Flyt_ind'], inplace=True)
        merged_admissions = []
        current_row = None
        for index, row in adm.iterrows():
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
                           'ADMISSION': admission['Flyt_ind'], 
                           'DISCHARGE': admission['Flyt_ud']})
        final_df = pd.DataFrame(events)
        final_df['ADMISSION_ID'] = self.assign_hash(final_df)

        return final_df
    
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
        file_path = join(self.dump_path, cfg.filename) if self.dump_path is not None else cfg.filename
        print(file_path)
        ds = Dataset.Tabular.from_parquet_files(path=(self.datastore,file_path))
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
            if mode == 'w':
                df.to_csv(path, index=True, mode=mode)
            else: 
                df.to_csv(path, index=True, mode=mode, header=False)
        else:
            raise ValueError(f"Filetype {file_type} not implemented yet")
