from src.logger import logging
from src.exception import CustomException
import os,sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataInjectionConfig:
    raw_data_path:str = os.path.join('artifacts','raw_data.csv')
    train_data_path:str = os.path.join('artifacts','train_data.csv')
    test_data_path:str = os.path.join('artifacts','test_data.csv')


class DataInjection:
    def __init__(self) -> None:
        
        self.data_path = DataInjectionConfig()

    def initiate_data_injection(self):
        try:
            logging.info('Initiating Data Injection')

            #Read the data for the source
            df = pd.read_csv('notebook\data\concrete_data.csv')

            os.makedirs(os.path.dirname(self.data_path.raw_data_path),exist_ok=True)
            df.to_csv(self.data_path.raw_data_path,index=False,header=True)

            logging.info("Train Test split ")
            train_data,test_data = train_test_split(df,test_size=0.2,random_state=42)
            train_data.to_csv(self.data_path.train_data_path,index=False,header=True)
            test_data.to_csv(self.data_path.test_data_path,index=False,header=True)
            logging.info('Data Injection Complited')
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj = DataInjection()
    obj.initiate_data_injection()

