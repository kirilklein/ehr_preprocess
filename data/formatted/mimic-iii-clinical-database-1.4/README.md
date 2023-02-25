# Preprocessed Files
- patients_info.parquet: (use last admission to get the info)
    - PID, GENDER, BIRTHDATE, DEATHDATE, INSURANCE, LANGUAGE, RELIGION, MARITAL_STATUS, ETHNICITY
- concept.lab.parquet: 
    - LOINC system, if missing description is usedW
- concept.med.parquet:
    - NDC and GSN codes available, we take the drugname and will map it using drugbank later

Events with missing TIMESTAMP are removed

