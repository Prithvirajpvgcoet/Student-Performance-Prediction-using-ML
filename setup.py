from setuptools import find_packages, setup 
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_pth:str)->List[str]:
   '''this function will return the list of requirements'''
   with open(file_pth) as file_obj:
     req=file_obj.readlines()
     req=[req.replace("\n","") for req in req]
     if HYPEN_E_DOT in req:
         req.remove(HYPEN_E_DOT)
    
   return req
setup(    name='mlproject',
    version='0.0.1',
    author='prithviraj',
    author_email='prithvirajthorat0770@gmail.com',
    packages=find_packages(),
    install_requires=['pandas', 'numpy', 'seaborn']
    )