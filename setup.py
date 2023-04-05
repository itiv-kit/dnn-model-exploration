from setuptools import find_packages, setup

setup(
    name="model_explorer",
    install_requires=[
        'pymoo',
        'torch',
        'torchvision',
        'webdataset',
        'matplotlib',
        'tqdm',
        'pandas',
        'seaborn',
        'gitpython',
        'opencv-python',
        'timm',
        'visdom'
    ],
    version='0.3',
    packages=find_packages(),
)
