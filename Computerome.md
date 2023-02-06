# File overview
#### AdministreretMedicin
> Key.Patient
> Key.Indlæggelse
> TAKEN_TIME
> ATC
> ADM_DOSIS

#### ADT_haendelser.csv
> Key.Patient
> Key.Indlæggelse
> Aktionsdiagnosekode
> Hændelsestidspunkt.Dato.tid
> Hændelsesansvarlig.Hospital.navn
> Hændelsesansvarlig.OVerafdeling.navn
> Hændelsesansvarlig.Afdeling.navn
> Hændelsesansvarlig.Afsnit.navn
> Kontakt.startdato.Dato.tid
> Kontakt.slutdato.Dato.tid
> Hændelsestype.navn
> Patient.alder.ved.Behandlingskontaktens.start

#### Aktive_problemliste_diagnoser
> Key
> CURRENT_ICD10_LIST
> NOTED_DATE

#### AlleProevesvar.csv
> Key.Patient
> COMPONENT
> EXTERNAL_NAME
> SPECIMN_TAKEN_TIME
> RESULT_TIME
> ORD_VALUE

#### Covid.19_positive_proevesvar
> Key.Patient
> X2 (Indtastest)
> Key.Indlæggelse
> X4 (Final results)
> Prøve.bestilt
> Prøvesvar.modtaget
> LAB_RESULT


#### Indlaeggelser_og_diagnoser
> Key.Indlæggelse
> DiagnoseType
> SKSKode

16317 Key.Indlæggelse not found in other files

Found 41088 entries where Key.Indlæggelse exists in other dataframes without the SKSKode present
21459 unique admissions

##### Top 10 codes that can be added
| SKSKode  | #Num  | Description                                                                |
|----------|-------|----------------------------------------------------------------------------|
| DB342A   | 15050 | COVID-19-infektion uden angivelse af lokalisation                          |
| DB972A   | 1869  | COVID-19 svær akut respiratorisk syndrom                                   |
| DZ038PA1 | 1070  | Observation pga. mistanke om COVID-19-infektion                            |
| DZ348A   | 869   | Graviditet, flergangsfødende                                               |
| DZ340    | 786   | Gravidetet, førstegangsfødende                                             |
| DI109    | 710   | Essentiel hypertension                                                     |
| DN189    | 514   | Kronisk nyreinsuffciens UNS                                                |
| DO985    | 465   | Anden viral sygdom som komplicerer graviditet, fødsel eller barselsperiode |
| DB972    | 423   | Coronavirus som årsag til sygdom                                           |
| DR060    | 418   | Dyspnø                                                                     |



ITA_ophold_og_respirator.csv
> Key.Patient
> REGION_NAME
> OVERAFDELING_NAME
> Aktionsdiagnose
> ICU_STAY_START
> ICU_STAY_END
> RespiratorStart
> RespiratorEnd

OrdineretMedicin
> Key.Patient
> ORDER_START_TIME
> ORDER_END_TIME
> NAME
> HV__DISCRETE_DOSE
> ATC
> ATC5
> ATC4

Patienter_og_Koen
> Key.Patient
> Gender
> Weight
> Height
> BMI

Patienters_evt_doedsdato.csv
> Key.Patient
> Patient.dødsdato.Dato.tid

Patienter_og_indlæggelser
> Key.Patient
> Key.Indlæggelse
> Aktionsdiagnosekode
> Bidiagnosekode
> Kontakt.startdato.Dato.tid
> Kontakt.slutdato.Dato.tid

Tobak_og_Alkohol
> Key.Patient
> Timestamp
> Smoker
> PakkerOmDagen
> Drikker
> Vaper
> SortingPåDato

VitalSigns:
> Key.Patient
> DisplayName
> RECORDED_TIME
> MEAS_VALUE_clean
> NumericValue
