from code.utilities import run_preprocessing
from code.model import housing_price_predictor


preprocess_data = True


if __name__ == '__main__':
    if preprocess_data:
        run_preprocessing('./code/data_exploration_and_processing.ipynb')
    predictor = housing_price_predictor()
    predictor.split_data()
    predictor.train()
    predictor.predict()
