from setuptools import find_packages, setup

setup(
    name="model_explorer",
    install_requires=[
        'pymoo',
        'torch>=1.13.1',
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
    version='1.0',
    description="A tool to support automatic exploration on how DNN models react to quantization and artificially introduced sparsity",
    # url="https://github.com/itiv-kit/dnn-model-exploration",
    packages=find_packages(),
)
