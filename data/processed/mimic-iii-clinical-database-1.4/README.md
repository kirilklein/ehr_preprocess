Preprocessing Description:
- concept.lab.parquet: 
    - LOINC system, if missing description is used
    - Labevents have a continuous or categorical outcome. 
    - If categorical, assign int 0, 1, 2, 3, ... to VALUE, set VALUE_UNIT to 'categorical', assign name to VALUE_CAT
    - New unseen values can get the value max_num+1 assigned
    
- concept.diag.parquet:
    - ICD9
    - No timestamps given for diagnosis, take AMDIT_TIME (hospital admission time) instead
- concept.med.parquet:
    - NDC and GSN codes available, we take the drugname and will map it using drugbank later
    - STARTDATE and ENDDATE available, take STARTDATE as TIMESTAMP
    - DOSE_VAL_RX and DOSE_UNIT_RX are used as VALUE and UNIT
    - If categorical, assign int 0, 1, 2, 3, ... to VALUE, set VALUE_UNIT to 'categorical', assign name to VALUE_CAT
    - FORM_UNIT_DISP might also be relevent (CAP, TAB, SYR, ml...)
