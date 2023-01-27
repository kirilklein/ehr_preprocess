from os.path import dirname, join, realpath

base_dir = dirname(dirname(dirname(realpath(__file__))))

class BasePreprocessor:
    def __init__(self, cfg):
        self.cfg = cfg
        if not cfg.data.raw_data_path is None:
            self.raw_data_path = cfg.data.raw_data_path
        else:
            self.raw_data_path = join(base_dir, 'data', 'raw', 'mimic-iii-clinical-database-1.4')
    def forward(self):
        raise NotImplementedError


class MimicIII(BasePreprocessor): 
    def __init__(self, cfg):
        super(MimicIII, self).__init__(cfg)
    
    def forward(self):
        print('Preprocess MimicIII from: ', self.raw_data_path)
        
