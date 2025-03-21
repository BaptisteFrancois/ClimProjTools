from setuptools import setup, find_packages

setup(
    name='ClimProjTools',
    version='0.0.1',
    description='A set of tools for downloading and processing CMIP6 data from the CDS',
    url='https://github.com/BaptisteFrancois/ClimProjTools.git',
    author='BaptisteFrancois',
    author_email='BaptisteFrancois51@gmail.com',
    license='MIT',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'pandas',
        'seaborn',
        'geopandas',
        'cdsapi',
        'xarray',
        'shapely'
    ],
    keywords=['python', 'Climate Change', 'CMIP6', 'CDS'],
)