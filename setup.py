from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path:str)->List[str]:
    """
        Args:
            file_path (str): _description_
        Returns:
            List[str]: _description_
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
    
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name = 'mlproject',
    version = '0.0.1',
    author = 'thehrmwn',
    author_email = 'adityahh.rm@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)