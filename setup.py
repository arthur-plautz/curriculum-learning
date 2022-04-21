import re
from setuptools import setup, find_packages

requirements_list = open('./requirements.txt', 'r').read()

setup(
    name="cl_models",
    version="0.1.6",
    packages=find_packages(
        include=[
            "cl_models",
            "cl_models.specialist",
            "cl_models.specialist.*",
            "cl_models.generator",
            "cl_models.generator.*",
        ]
    ),
    install_requires=requirements_list
)