import pandas as pd
import os,sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from src.utils import saveobject
from sklearn.ensemble import GradientBoostingRegressor

from src.components.data_transformation import DataTransformation
from src.components.data_injection import DataInjection

@dataclass
class ModelTranerConfig:
    trained_model_file_path:str = os.path.join('aritfacts','model.pkl')

class ModelTrander:
    def __init__(self) -> None:
        self.model_traned_config = ModelTranerConfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            #train test split
            logging.info('Train Test split for model traning')
            X = train_arr[:,:-1]
            y = train_arr[:-1]

            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
            logging.info('Initilizing and tuning the model')
            gf = GradientBoostingRegressor(n_estimators=400,learning_rate=0.2,max_features='sqrt',subsample=1.0)
            gf.fit(X_train,y_train)
            y_pred = gf.predict(X_test)
            score = r2_score(y_test,y_pred)
            logging.info(f'The accuracy of the model is{score}')

            saveobject(file_path=self.model_traned_config.trained_model_file_path,obj=gf)
           
        except Exception as e:
            raise CustomException(e,sys)



