import pandas as pd

def format_diagnosis(diag, cfg):
    # Search code in diagnoses. If there is no diagnosis code, use the diagnosis extracted from the text
    diag['code'] = diag['Diagnose'].str.extract(r'\((D.*?)\)', expand=False)
    if 'fill_diags' in cfg and cfg['fill_diags']:
        diag['code'] = diag['code'].fillna(diag['Diagnose'])
    diag['CONCEPT'] = diag.Diagnosekode.fillna(diag.code)
    diag = diag.drop(['code', 'Diagnose', 'Diagnosekode'], axis=1)
    diag = diag.rename(columns={'CPR_hash':'PID', 'Noteret_dato':'TIMESTAMP'})
    return diag

def format_procedure(proc, cfg):
    proc['CONCEPT'] = proc['ProcedureCode'].str.replace(' ', '')
    proc = proc.drop(['ProcedureCode'], axis=1)
    proc = proc.rename(columns={'CPR_hash':'PID', 'ServiceDatetime':'TIMESTAMP'})
    proc['CONCEPT'] = proc['CONCEPT'].map(lambda x: 'P'+x)
    return proc

def format_labtest(labs, cfg):
    labs = labs.rename(columns={'CPR_hash':'PID', 'BestOrd':'CONCEPT', 'Bestillingsdato': 'TIMESTAMP', 'Resultatværdi':'RESULT'})
    labs['CONCEPT'] = labs['CONCEPT'].map(lambda x: 'LAB_'+x)
    return labs

def format_medication(med, cfg):
    med.loc[:, 'CONCEPT'] = med.ATC.fillna('Ordineret_lægemiddel')
    med.loc[:, 'TIMESTAMP'] = med.Administrationstidspunkt.fillna("Bestillingsdato")
    med = med.rename(columns={'CPR_hash':'PID'})
    med = med[['PID','CONCEPT','TIMESTAMP']]
    med['CONCEPT'] = med['CONCEPT'].map(lambda x: 'M'+x)
    return med

def format_register_diagnosis(diag, cfg, forl, kont, mapping):
    def add_forl_diag(df, forl):
        commons = pd.merge(df[['PID']], forl[['CPR_hash']], left_on='PID', right_on='CPR_hash')
        forl_filtered = forl[forl['CPR_hash'].isin(commons['CPR_hash'])]
        merged = pd.merge(forl_filtered, df, left_on=['CPR_hash', 'henvisningsaarsag'], right_on=['PID', 'CONCEPT'], how='left', indicator=True)
        mask = merged['_merge'] == 'left_only'
        new_rows = forl_filtered[mask]
        
        if new_rows.empty:
            return df

        new_rows = new_rows.rename(columns={'CPR_hash': 'PID', 'henvisningsaarsag': 'CONCEPT', 'TIMESTAMP_START': 'TIMESTAMP'})
        new_rows = new_rows.loc[:, ['PID', 'CONCEPT', 'TIMESTAMP']]
        exploded_df = pd.concat([df, new_rows], ignore_index=True)
        return exploded_df

    diag['dw_ek_kontakt'] = diag['dw_ek_kontakt'].astype(int)
    kont['dw_ek_kontakt'] = kont['dw_ek_kontakt'].astype(int)

    merged_df = pd.merge(
        diag, 
        kont, 
        on="dw_ek_kontakt", 
        how="inner"
    )
    merged_df = merged_df.rename(columns={'CPR_hash':'PID', 'diagnosekode':'CONCEPT', 'TIMESTAMP_START':'TIMESTAMP', })
    if cfg.add_details:
        merged_df = add_forl_diag(merged_df, forl)
    
    merged_df = merged_df.loc[:, ['PID', 'CONCEPT', 'TIMESTAMP']]
    return merged_df

def format_register_medication(med, cfg, forl, kont, mapping):
    merged_df = pd.merge(
        med, 
        mapping, 
        on='PID',
        how='inner'
    ).drop(['PID'], axis=1)
    merged_df['TIMESTAMP'] = pd.to_datetime(merged_df['eksd'])
    merged_df = merged_df.rename(columns={'atc':'CONCEPT', 'CPR_hash':'PID', 'vnr':'VNR'})
    merged_df['CONCEPT'] = merged_df['CONCEPT'].map(lambda x: 'M'+x)

    if cfg.add_symptoms: 
        indo_df = merged_df[merged_df['indo'] != 'nan'].copy()
        indo_df['CONCEPT'] = 'S' + indo_df['indo'].astype(float).astype(int).astype(str) # Convert to int to remove decimal point
        new_rows = indo_df.loc[:, ['PID', 'CONCEPT', 'TIMESTAMP', 'VNR']]
        merged_df = pd.concat([merged_df, new_rows], ignore_index=True)
            
    merged_df = merged_df.loc[:, ['PID', 'CONCEPT', 'TIMESTAMP', 'VNR']]
    return merged_df

def format_register_procedures(proc, cfg, forl, kont, mapping):
    proc['dw_ek_kontakt'] = proc['dw_ek_kontakt'].astype(int)
    kont['dw_ek_kontakt'] = kont['dw_ek_kontakt'].astype(int)

    merged_df = pd.merge(
        proc, 
        kont, 
        on="dw_ek_kontakt", 
        how="inner",
        suffixes=('_proc', '_kont')
    )
    merged_df['TIMESTAMP'] = pd.to_datetime(merged_df['dato_start_proc'].astype(str) + ' ' + merged_df['tidspunkt_start_proc'].astype(str))
    merged_df = merged_df.rename(columns={'CPR_hash':'PID', 'procedurekode':'CONCEPT'})    
    merged_df = merged_df.loc[:, ['PID', 'CONCEPT', 'TIMESTAMP']]
    return merged_df

def format_register_procedures_surgical(proc, cfg, forl, kont, mapping): 
    return format_register_procedures(proc, cfg, forl, kont, mapping)

def format_register_procedures_non_surgical(proc, cfg, forl, kont, mapping): 
    return format_register_procedures(proc, cfg, forl, kont, mapping)


