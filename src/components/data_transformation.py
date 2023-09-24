from src.logger import logging
from src.exception import CustomException
import os,sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.utils import saveobject
#module for data transformation
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#get the data from data injection
from src.components.data_injection import DataInjection

@dataclass
class DataTransformationConfig:
    data_transformation_file_config:str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.preprocessor_file_path = DataTransformationConfig()


    def get_preprocessor_obj(self,train_data,test_data):
        try:
            X = train_data.drop(['Strength'],axis=1)
            y = train_data['Strength']
            num_col = X.select_dtypes(include=[int, float]).columns

            # Define the transformer for numerical columns 
    
            logging.info('Data Pipeline initiated')
            preprocessor = ColumnTransformer(
                transformers=[
                    ('standard_scaler', StandardScaler(), num_col)
                ],
                remainder='passthrough'
            )

            logging.info('Returning preprocessor obj')
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def intitiate_data_transformation(self,train_data_path,test_data_path):
        try:
            target_col = ['Strength']
            
            #get both train and test data
            logging.info('Reading the dataset')
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path) 
            
            logging.info('Getting the preprocessor obj')
            preprocessor_obj = self.get_preprocessor_obj(train_data,test_data)

            #seperate the input and output feature for Traning data
            logging.info("Spliting the Training Data set")
            input_feature_train_df = train_data.drop(columns=target_col,axis=1)
            target_feature_train_df = train_data[target_col]

            logging.info('Spliting the Test Dataset')
            input_feature_test_df = test_data.drop(columns=target_col,axis=1)
            target_feature_test_df = test_data[target_col]

            #preform the transformation
            logging.info("Transforming the traning and testing set")
            input_feature_train_transformed = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_transformed = preprocessor_obj.transform(input_feature_test_df)

            #target_feature_train_transformed = preprocessor_obj.transform(target_feature_train_df)
            #target_feature_test_transformed = preprocessor_obj.transform(target_feature_test_df)

            train_arr = np.c_[input_feature_train_transformed,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_transformed, np.array(target_feature_test_df)]

            saveobject(file_path=self.preprocessor_file_path.data_transformation_file_config,
                       obj=preprocessor_obj)
            logging.info('Data Transformation complited')
            return(train_arr,test_arr)
            
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj_data_injection = DataInjection()
    train_data,test_data = obj_data_injection.initiate_data_injection()
    obj = DataTransformation()
    obj.intitiate_data_transformation(train_data,test_data)


