from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name="credit_scoring_model",
    version="0.0.1",
    author="Anmol",
    author_email="anmolgg0001@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)