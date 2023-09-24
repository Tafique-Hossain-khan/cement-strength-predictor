
import pickle
import os,sys
from src.exception import CustomException
def saveobject(file_path,obj):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,'wb') as f:
            pickle.dump(obj,f)

    except Exception as e:
        raise CustomException(e,sys)

