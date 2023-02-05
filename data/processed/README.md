Preprocessing Description:
- concept.lab.parquet: 
    - LOINC system, if missing description is used
    - Labevents have a continuous or categorical outcome. 
    - If categorical, assign int 0, 1, 2, 3, ... to VALUE, set VALUE_UNIT to 'categorical', assign name to VALUE_CAT
    - New unseen values can get the value max_num+1 assigned
    
- concept.diag.parquet:
    - No timestamps