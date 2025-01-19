  @staticmethod
    def format_diagnosis(diag, cfg):
        # Search code in diagnoses. If there is no diagnosis code, use the diagnosis extracted from the text
        diag['code'] = diag['Diagnose'].str.extract(r'\((D.*?)\)', expand=False)
        if 'fill_diags' in cfg and cfg['fill_diags']:
            diag['code'] = diag['code'].fillna(diag['Diagnose'])
        diag['CONCEPT'] = diag.Diagnosekode.fillna(diag.code)
        diag = diag.drop(['code', 'Diagnose', 'Diagnosekode'], axis=1)
        diag = diag.rename(columns={'CPR_hash':'PID', 'Noteret_dato':'TIMESTAMP'})
        return diag

    @staticmethod
    def format_procedure(proc, cfg):
        proc['CONCEPT'] = proc['ProcedureCode'].str.replace(' ', '')
        proc = proc.drop(['ProcedureCode'], axis=1)
        proc = proc.rename(columns={'CPR_hash':'PID', 'ServiceDatetime':'TIMESTAMP'})
        proc['CONCEPT'] = proc['CONCEPT'].map(lambda x: 'P'+x)
        return proc

    @staticmethod
    def format_labtest(labs, cfg):
        labs = labs.rename(columns={'CPR_hash':'PID', 'BestOrd':'CONCEPT', 'Bestillingsdato': 'TIMESTAMP', 'Resultatværdi':'RESULT'})
        labs['CONCEPT'] = labs['CONCEPT'].map(lambda x: 'LAB_'+x)
        return labs
    
    @staticmethod
    def format_medication(med, cfg):
        med.loc[:, 'CONCEPT'] = med.ATC.fillna('Ordineret_lægemiddel')
        med.loc[:, 'TIMESTAMP'] = med.Administrationstidspunkt.fillna("Bestillingsdato")
        med = med.rename(columns={'CPR_hash':'PID'})
        med = med[['PID','CONCEPT','TIMESTAMP']]
        med['CONCEPT'] = med['CONCEPT'].map(lambda x: 'M'+x)
        return med
