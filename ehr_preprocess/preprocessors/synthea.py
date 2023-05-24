from base import BasePreprocessor

class SyntheaPrepocessor(BasePreprocessor):
    # __init__ is inherited from BasePreprocessor
    pass


if __name__ == '__main__':
    from hydra import initialize, compose
    with initialize(config_path="../../configs"):
        cfg = compose(config_name="synthea.yaml")
    preprocessor = SyntheaPrepocessor(cfg)
    preprocessor()
