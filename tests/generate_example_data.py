import pandas as pd
import numpy as np
import random
import hashlib

def generate_cpr_hash(n):
    hashes = []
    for _ in range(n):
        random_str = str(random.randint(1000000000, 9999999999))
        hash_object = hashlib.sha256(random_str.encode())
        hashes.append(hash_object.hexdigest())
    return

def generate_medical_code(n, prefix='D'):
    return [f"{prefix}{random.randint(100, 999)}" for _ in range(n)]

def generate_diagnosis(n=1000, seed=0):
    words = ["syndrome", "infection", "disorder", "acute", "chronic", "complication", "neoplasm", 
             "unspecified", "degeneration", "injury", "abnormality", "malformation", "failure", "collapse"]
    body_parts = ["lung", "brain", "heart", "kidney", "liver", "nerve", "muscle", "stomach", "skin", "eye", "ear"]
    
    diagnoses = []
    for _ in range(n):
        phrase = f"{random.choice(words)} of the {random.choice(body_parts)}"
        if random.random() > 0.3:  # 70% chance to include (Dxxx)
            diagnoses.append(f"{phrase} ({generate_medical_code(1, "D")[0]})")
        else:
            diagnoses.append(phrase)
    return diagnose

def generate_diagnosis(n=1000, seed=0):
    np.random.seed(seed)
    df = pd.DataFrame({
        'CPR_hash': np.random.randint(0, 100, n),
        'Diagnosekode': np.random.choice(['A', 'B', 'C'], n),
        'Diagnose': np.random.choice(['A', 'B', 'C'], n),
        'Noteret_dato': pd.date_range('2020-01-01', periods=n, freq='D'),
        'Løst_dato': pd.date_range('2020-01-01', periods=n, freq='D')
    })


    df = pd.DataFrame({
        'dw_ek_kontakt': np.random.randint(0, 100, n),
        'CPR_hash': np.random.randint(0, 100, n),
        'diagnosekode': np.random.choice(['A', 'B', 'C'], n),
        'TIMESTAMP_START': pd.date_range('2020-01-01', periods=n, freq='D')
    })
    return df


CPR_hash	Diagnosekode	Diagnose	Noteret_dato	Løst_dato
0	0000196717E9D914CD43F0DDE29DDC111DA89D1F1A545C...	DH259	Aldersbetinget grå stær (>=50 år) UNS (DH259)	2017-10-27	NaT
1	0000196717E9D914CD43F0DDE29DDC111DA89D1F1A545C...	DZ018B	Adenomkontrolprogram, tidl. fund af mellem/høj...	2019-08-22	NaT
2	0000196717E9D914CD43F0DDE29DDC111DA89D1F1A545C...	DJ189	Pneumoni UNS (DJ189)	2021-12-04	NaT
3	0000196717E9D914CD43F0DDE29DDC111DA89D1F1A545C...	DE119A	Type 2-diabetes UNS (DE119A)	2021-12-12	NaT
4	0000196717E9D914CD43F0DDE29DDC111DA89D1F1A545C...	DI109	Essentiel hypertension (DI109)	2021-12-12	NaT



def format_diagnosis(diag, cfg):
    # Search code in diagnoses. If there is no diagnosis code, use the diagnosis extracted from the text
    diag['code'] = diag['Diagnose'].str.extract(r'\((D.*?)\)', expand=False)
    if 'fill_diags' in cfg and cfg['fill_diags']:
        diag['code'] = diag['code'].fillna(diag['Diagnose'])
    diag['CONCEPT'] = diag.Diagnosekode.fillna(diag.code)
    diag = diag.drop(['code', 'Diagnose', 'Diagnosekode'], axis=1)
    diag = diag.rename(columns={'CPR_hash':'PID', 'Noteret_dato':'TIMESTAMP'})
    return diag

def format_diagnosis(diag, cfg):
    # Search code in diagnoses. If there is no diagnosis code, use the diagnosis extracted from the text
    diag['code'] = diag['Diagnose'].str.extract(r'\((D.*?)\)', expand=False)
    if 'fill_diags' in cfg and cfg['fill_diags']:
        diag['code'] = diag['code'].fillna(diag['Diagnose'])
    diag['CONCEPT'] = diag.Diagnosekode.fillna(diag.code)
    diag = diag.drop(['code', 'Diagnose', 'Diagnosekode'], axis=1)
    diag = diag.rename(columns={'CPR_hash':'PID', 'Noteret_dato':'TIMESTAMP'})
    return diag