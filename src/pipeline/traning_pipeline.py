from src.components.data_transformation import DataTransformation
from src.components.data_injection import DataInjection
from src.logger import logging
import os,sys
from src.exception import CustomException
from src.components.model_traner import ModelTraner



if __name__ == "__main__":
    try:
        obj_data_injection = DataInjection()
        train_data,test_data = obj_data_injection.initiate_data_injection()
        obj_data_transformation = DataTransformation()
        train_arr,test_arr = obj_data_transformation.intitiate_data_transformation(train_data,test_data)
        logging.info('Model Training initiated')
        obj_model_trainer = ModelTraner()
        obj_model_trainer.initiate_model_training(train_arr,test_arr)
        
    except Exception as e:
        raise CustomException(e,sys)