from setuptools import find_packages,setup
from typing import List

def get_packages(file_path:str)->List[str]:
    
    requirnments =[]
    with open(file_path) as f:
        requirnments = f.readlines()
        requirnments = [req.replace('\n','')for req in requirnments]

        if "-e ." in requirnments:
            requirnments.remove('-e .')
    return requirnments

setup(
    name="ML Project",
    version='0.0.1',
    author='Tafique',
    author_email='tafiquehossain2003@gmail.com',
    packages= find_packages(),
    install_requires = get_packages('requirnment.txt'),

)