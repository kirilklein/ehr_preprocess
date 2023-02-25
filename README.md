# ehr_preprocess
This repo can be used to produce formatted versions of Electronic Health Records for MIMIC and other datasets.<br>
Data is separated into types:
- patients_info: PID and corresponding static information
- concept.{eventtype}: PID, ADMISSION_ID, CONCEPT, TIMESTAMP, (TIMESTAMP_END, VALUE, VALUE_UNIT, ICUSTAY_ID)
