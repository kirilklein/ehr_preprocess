# ehr_preprocess (archived)

> **Superseded by [FGA-DIKU/ehr2meds](https://github.com/FGA-DIKU/ehr2meds).**
> ehr2meds covers everything here (raw EHR dump → normalized concept tables) and additionally converts to the [MEDS](https://github.com/Medical-Event-Data-Standard/meds) format. Use it instead.

This repo produced flat concept tables from Danish hospital EHR (SP) and register dumps:

- `patients_info`: PID + static information
- `concept.{eventtype}`: PID, ADMISSION_ID, CONCEPT, TIMESTAMP, (TIMESTAMP_END, VALUE, VALUE_UNIT)

It is no longer maintained.
