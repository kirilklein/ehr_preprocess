# EHR Preprocess

This repository provides a tool to format version of Electronic Health Records (EHR) to be compatible with [CORE-BEHRT](https://github.com/mikkelfo/CORE-BEHRT/tree/azure)[1].

## Example Dataset
A small example dataset generated with [synthea](git@github.com:synthetichealth/synthea.git)[2] is provided in the `example_data` directory.

## Configuration
Specify the config file in ehr_preprocess/main.py and run the script to generate the formatted data.<br>

## Output
The formatted data will be saved in the `output` directory in the following format:
- patients_info: PID and corresponding static information (e.g. sex, ethnicity etc.)
- concept.{eventtype}: PID, ADMISSION_ID, CONCEPT, TIMESTAMP, (TIMESTAMP_END, VALUE, VALUE_UNIT, ICUSTAY_ID) holding medical events.

## References
[1] Odgaard, Mikkel, et al. "CORE-BEHRT: A Carefully Optimized and Rigorously Evaluated BEHRT." MLHC 2024.
[2] Walonoski, Jason, et al. "Synthea: An approach, method, and software mechanism for generating synthetic patients and the synthetic electronic health care record." Journal of the American Medical Informatics Association 25.3 (2018): 230-238.
 