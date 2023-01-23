from setuptools import find_packages, setup

setup(
    name="model_explorer",
    install_requires=[
        'pymoo',
        'torch',
        'matplotlib'
    ],
    version='0.3',
    packages=find_packages(),
)
