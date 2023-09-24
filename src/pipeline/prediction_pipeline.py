
import os,sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import loadobject

#need to read get the obj and then train the custom data
class PredictionPipeline:
    def __init__(self) -> None:
        pass

    def predict(self,feature):
        try:
            logging.info('Getting the file path')
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            logging.info('Getting the preprocessor and model object')
            preprocessor = loadobject(preprocessor_path)
            model = loadobject(model_path)
            logging.info('Scaling and predicting')
            data_scaled = preprocessor.transform(feature)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,cement,slag,ash,water,Superplasticizer,
                 coarse_agg,fine_agg,age) -> None:
        self.cement = cement
        self.slag = slag
        self.water = water 
        self.ash = ash
        self.Superplasticizer = Superplasticizer
        self.coarse_agg = coarse_agg
        self.fine_agg = fine_agg
        self.age = age

    def get_custom_data(self):
        try:
            data = {
                'cement': [self.cement],
                'slag': [self.slag],
                'ash': [self.ash],
                'water':[self.water],
                'Superplasticizer': [self.Superplasticizer],
                'coarse_agg': [self.coarse_agg],
                'fine_agg': [self.fine_agg],
                'age': [self.age]
                }
            logging.info(f'{pd.DataFrame(data=data)}')
            return pd.DataFrame(data=data)
        except Exception as e:
            raise CustomException(e,sys)
        

