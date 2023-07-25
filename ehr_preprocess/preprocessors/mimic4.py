from preprocessors import base

class MIMIC4Preprocessor(base.BasePreprocessor):
    """Extracts events from MIMIC-III database and saves them in a single file."""

    def __init__(self, cfg, test=False):
        super(MIMIC4Preprocessor, self).__init__(cfg)
        self.test = test

    def __call__(self):
        self.patients_info()
        
    def patients_info(self):
        # identical to base
        for key, cfg in self.config.patients_info.items():
            df = self.load_csv(cfg)
            
            self.set_patient_info(df)

        # Convert info dict to dataframe
        df = self.info_to_df()
        # get also info from admission and omr (weight, height, ...)

    def cacl_birthdates(self):
        pass
   
