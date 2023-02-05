Here we describe briefly the preprocessing:
lab.parquet: 
    Labevents have a continuous or categorical outcome. 
    For categorical outcomes, we assign integer values 0, 1, 2, 3, ... and set unit to 'categorical' 
    Unseen values can get the value max_num+1 assigned
    We don't perform conversion between systems
