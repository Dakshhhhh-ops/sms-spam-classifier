from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path:str)->List[str]:
    """
    This function will return the list of requirements mentioned in the file
    """
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        new_requirements=[]
        for i in requirements:
            new_requirements.append(i.replace("\n",""))

        if '-e .' in new_requirements:
            new_requirements.remove('-e .')
               
    return new_requirements
     

setup(
    name='chicken_disease_classifier',
    version='0.0.1',
    author='Daksh',
    author_email='dakshwadhwa2007@gmail.com',
    description='A machine learning project for classifying chicken diseases.',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)
