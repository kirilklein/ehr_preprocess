import pandas as pd
import numpy as np
import random
import hashlib
import argparse
import os
import string
    

from constants import (
    DESCRIPTIONS, BODY_PARTS, 
    MEDICATION_NAMES, MED_TYPES, MED_UNITS, MED_ADMINISTRATIONS, MED_INFUSION_SPEED, MED_INFUSION_DOSE, MED_ACTIONS, 
    PROCEDURE_NAMES,
    LAB_TESTS, LAB_RESULTS, LAB_ANTIBIOTICS, LAB_SENSITIVITIES, LAB_ORGANISMS
)

DEFAULT_N_CONCEPTS = 5 # concepts per patient
DEFAULT_N = 100 # number of patients
DEFAULT_WRITE_DIR = "example_data"

def generate_cpr_hash(n):
    hashes = []
    for _ in range(n):
        random_str = str(random.randint(1000000000, 9999999999))
        hash_object = hashlib.sha256(random_str.encode())
        hashes.append(hash_object.hexdigest())
    return hashes

def generate_medical_code(n, start=100, end=999, mix_letters=False, prefix=None):
    if prefix is None:
        prefix = ""

    codes = []
    for _ in range(n):
        number_part = str(random.randint(start, end))
        letter_part = "".join(random.choices(string.ascii_uppercase, k=2)) if mix_letters else ""
        codes.append(f"{prefix}{letter_part}{number_part}")

    return codes

def generate_timestamps(birthdates, deathdates, n=1000):
    birthdates = birthdates.astype(np.int64) // 10**9
    deathdates = deathdates.astype(np.int64) // 10**9
    random_timestamps = [np.random.randint(birthdate, deathdate) for birthdate, deathdate in zip(birthdates, deathdates)]
    timestamps = pd.to_datetime(random_timestamps, unit="s")
    return timestamps


def generate_diagnosis_csv(save_dir, hashes, birthdates, deathdates, seed=0):
    def generate_diagnosis(n=1000, diag_codes=None):        
        diagnoses = []
        if diag_codes is not None:
            n = len(diag_codes)
        else:
            diag_codes = generate_medical_code(n, "D")
        for i in range(n):
            phrase = f"{random.choice(DESCRIPTIONS)} of the {random.choice(BODY_PARTS)}"
            if random.random() > 0.3:  # 70% chance to include (Dxxx)
                diagnoses.append(f"{phrase} ({diag_codes[i]})")
            else:
                diagnoses.append(phrase)
        return diagnoses

    save_name = 'concept.diagnosis.csv'

    total_concepts = len(hashes)
    diag_codes = generate_medical_code(total_concepts, prefix='D')
    df = pd.DataFrame({
        'CPR_hash': hashes,
        'Diagnosekode': diag_codes,
        'Diagnose': generate_diagnosis(total_concepts, diag_codes),
        'Noteret_dato': generate_timestamps(birthdates, deathdates, total_concepts),
        'Løst_dato': generate_timestamps(birthdates, deathdates, total_concepts)
    })
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(f'{save_dir}/{save_name}', index=False)

def generate_medication_csv(save_dir, hashes, birthdates, deathdates, seed=0):
    def generate_medication_description(n=1000):
        dose = [random.randint(10, 1000) for _ in range(n)]
        unit = [random.choice(MED_UNITS) for _ in range(n)]
        generic_names = [random.choice(MEDICATION_NAMES) for _ in range(n)]
        med_type = [random.choice(MED_TYPES) for _ in range(n)]
        descs = [f"{name} {med_type} {dose} {unit}".upper() for name, med_type, dose, unit in zip(generic_names, med_type, dose, unit)]
        return descs, generic_names, dose, unit

    total_concepts = len(hashes)
    description, generic_names, dose, unit = generate_medication_description(total_concepts)
    med_codes = generate_medical_code(total_concepts, start=1000, end=9999, mix_letters=True)
    df = pd.DataFrame({
        'CPR_hash': hashes,
        'BestOrd_ID': np.random.randint(1e9, 1e10, total_concepts),
        'Ordineret_lægemiddel': description,
        'Generisk_navn': generic_names,
        'ATC': med_codes,
        'Bestillingsdato': generate_timestamps(birthdates, deathdates, total_concepts),
        'Administrationstidspunkt': generate_timestamps(birthdates, deathdates, total_concepts),
        'Administrationsdosis': dose,
        'Dosisenhed': unit,
        'Administrationsvej': [random.choice(MED_ADMINISTRATIONS) for _ in range(total_concepts)],
        'Seponeringstidspunkt': generate_timestamps(birthdates, deathdates, total_concepts),
        'Ordineret_dosis': [random.randint(1, 500) for _ in range(total_concepts)],
        'Infusionshastighed': [random.choice(MED_INFUSION_SPEED) for _ in range(total_concepts)],
        'Infusionsdosis': [random.choice(MED_INFUSION_DOSE) for _ in range(total_concepts)],
        'Handling': [random.choice(MED_ACTIONS) for _ in range(total_concepts)]
    })
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(f'{save_dir}/concept.medication.csv', index=False)

def generate_procedure_csv(save_dir, hashes, birthdates, deathdates, seed=0):
    total_concepts = len(hashes)
    dates = pd.Series(generate_timestamps(birthdates, deathdates, total_concepts))
    df = pd.DataFrame({
        'CPR_hash': hashes,
        'ProcedureCode': generate_medical_code(len(hashes), start=100, end=999, mix_letters=True),
        'ProcedureName': [random.choice(PROCEDURE_NAMES) for _ in range(total_concepts)],
        'ServiceDate': dates.dt.date,
        'ServiceTime': dates.apply(lambda x: pd.Timestamp("1970-01-01") + pd.Timedelta(hours=x.hour, minutes=x.minute, seconds=x.second)),
        'ServiceDatetime': dates
    })
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(f'{save_dir}/concept.procedure.csv', index=False)

def generate_labtest_csv(save_dir, hashes, birthdates, deathdates, seed=0):
    total_concepts = len(hashes)
    dates = pd.Series(generate_timestamps(birthdates, deathdates, total_concepts))
    time_for_results = np.random.randint(0, 4, total_concepts)
    results_date = dates + pd.to_timedelta(time_for_results, unit='d')
    df = pd.DataFrame({
        'CPR_hash': hashes,
        'BestOrd_ID': np.random.randint(1e9, 1e10, total_concepts),
        'BestOrd': [random.choice(LAB_TESTS) for _ in range(total_concepts)],
        'Bestillingsdato': dates.dt.date,
        'Prøvetagningstidspunkt': dates,
        'Resultatdato': results_date,
        'Resultatværdi': [random.choice(LAB_TESTS) for _ in range(total_concepts)],
        'Antibiotika': [random.choice(LAB_ANTIBIOTICS) for _ in range(total_concepts)],
        'Følsomhed': [random.choice(LAB_SENSITIVITIES) for _ in range(total_concepts)],
        'Organisme': [random.choice(LAB_ORGANISMS) for _ in range(total_concepts)]
    })
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(f'{save_dir}/concept.labtest.csv', index=False)

def generate_patients_info(n_patients):
    # Set a random seed for reproducibility (optional)
    np.random.seed(42)

    # Define the range of birthdates (e.g., between 1940 and 2020)
    start_birthdate = np.datetime64("1970-01-01")
    end_birthdate = np.datetime64("2020-01-01")

    # Generate random birthdates between start and end dates
    birthdates = np.random.choice(
        np.arange(start_birthdate, end_birthdate, dtype="datetime64[D]"), n_patients
    )

    # Generate deathdates where some people are still alive (i.e., deathdate is NaT)
    death_prob = np.random.rand(n_patients)

    # For those with death_prob > 0.8, generate a deathdate between their birthdate and a future date (e.g., 2025)
    deathdates = np.where(
        death_prob > 0.8,
        np.array(
            [
                np.random.choice(
                    np.arange(
                        birthdate + np.timedelta64(10),
                        np.datetime64("2024-01-01"),
                        dtype="datetime64[D]",
                    )
                )
                for birthdate in birthdates
            ]
        ),
        pd.NaT,
    )
    genders = np.random.choice(["Mand", "Kvinde"], size=n_patients)

    # Generate random PIDs
    hashes = generate_cpr_hash(n_patients)

    return pd.DataFrame(
        {
            "CPR_hash": hashes,
            "Fødselsdato": birthdates,
            "Dødsdato": deathdates,
            "Køn": genders,
        }
    )


def generate_forloeb(mapping_merged):
    pids = mapping[[mapping['forloeb'] == True]][['PID']]
    
    pids_merged = pd.merge(pids, patient_info, on='CPR_hash')
    mask = pids_merged['Fødselsdato'].isna()
    pids_merged[mask, 'Fødselsdato'] = np.random.choice(np.arange(start_birthdate, end_birthdate, dtype="datetime64[D]"), n_patients)
    
    forloeb = pd.DataFrame({
        'CPR_hash': np.random.choice(mapping['CPR_hash'], size=n, replace=True),
        'forloeb': [random.choice([True, False]) for _ in range(n)],
        'forloeb_start': generate_timestamps(mapping['Fødselsdato'], mapping['Dødsdato'], n),

def generate_map_forl_kont(save_dir, hashes, patients_info, seed=0):
    pts_with_register_data = np.random.choice(hashes, size=len(hashes) // 2, replace=False)
    pts_with_epikur = np.random.choice(pts_with_register_data, size=len(pts_with_register_data) // 2, replace=False)
    pts_with_forl = np.random.choice(pts_with_register_data, size=(len(pts_with_epikur) // 5)*4, replace=False)

    # Create mapping
    pids = generate_cpr_hash(len(hashes) * 2)
    pids_mapping = {pid: cpr_hash for pid, cpr_hash in zip(pids[:len(hashes)], hashes)}
    mapping = pd.DataFrame({'PID': pids})
    mapping['CPR_hash'] = mapping['PID'].map(lambda x: pids_mapping.get(x, None))
    mapping['epikur'] = mapping['CPR_hash'].apply(lambda x: x in pts_with_epikur)
    mapping['kontakter'] = mapping['CPR_hash'].apply(lambda x: x in pts_with_forl)
    mapping['forloeb'] = mapping['CPR_hash'].apply(lambda x: x in pts_with_forl)

    for col in ['epikur', 'kontakter', 'forloeb']:
        mask = (mapping[col] == False) & (mapping['CPR_hash'].isna())
        mapping.loc[mask, col] = np.random.choice([True, False], size=mask.sum())

    mapping['t_adm'] = [np.random.choice([True, False]) for _ in range(len(mapping))]
    mapping['t_tumor'] = [np.random.choice([True, False]) for _ in range(len(mapping))]

    # Add birthday to merged
    start_birthdate = np.datetime64("1940-01-01")
    end_birthdate = np.datetime64("2020-01-01")
    mapping_merged = pd.merge(mapping, patients_info, on='CPR_hash', how='left')
    mask = mapping_merged['Fødselsdato'].isna()
    mapping_merged.loc[mask, 'Fødselsdato'] = np.random.choice(
        np.arange(start_birthdate, end_birthdate, dtype="datetime64[D]"), 
        size=mask.sum()
    )
    mapping_merged['Dødsdato'] = mapping_merged['Dødsdato'].fillna(np.datetime64("2025-01-01"))

def main_write(n_patients=DEFAULT_N, n_concepts=DEFAULT_N_CONCEPTS, write_dir=DEFAULT_WRITE_DIR):
    np.random.seed(0)
    patients_info = generate_patients_info(n_patients)
    patients_info.to_csv(f'{write_dir}/patients_info.csv', index=False)

    # Getting lists for the CPR_hash, birthdate, and deathdate
    pids = patients_info['CPR_hash']
    hashes = np.tile(pids, n_concepts)
    birthdates = np.tile(patients_info['Fødselsdato'], n_concepts)
    deathdates = np.tile(patients_info['Dødsdato'], n_concepts)
    birthdates = pd.to_datetime(birthdates)
    deathdates = pd.to_datetime(deathdates).fillna(pd.Timestamp(year=2025, month=1, day=1))

    for birth, death in zip(birthdates, deathdates):
        assert birth < death, f"Birthdate {birth} is not before deathdate {death}"

    # generate_diagnosis_csv(write_dir, hashes, birthdates, deathdates)
    # generate_medication_csv(write_dir, hashes, birthdates, deathdates)
    # generate_procedure_csv(write_dir, hashes, birthdates, deathdates)
    # generate_labtest_csv(write_dir, hashes, birthdates, deathdates)
    generate_map_forl_kont(write_dir, hashes, patients_info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate data for performance testing and profiling."
    )
    parser.add_argument(
        "--n-concepts",
        type=int,
        default=DEFAULT_N_CONCEPTS,
        help=f"Number of concepts to generate per patient (default: {DEFAULT_N_CONCEPTS})",
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=DEFAULT_N,
        help=f"Number of patients to generate (default: {DEFAULT_N})",
    )
    parser.add_argument(
        "--write-dir",
        type=str,
        default=DEFAULT_WRITE_DIR,
        help=f"Directory to write output files (default: {DEFAULT_WRITE_DIR})",
    )

    args = parser.parse_args()
    main_write(args.n_patients, args.n_concepts, args.write_dir)