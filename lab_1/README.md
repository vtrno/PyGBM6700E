# SpineTools

Python package for GBM6700E course lab 1 at Polytechnique Montréal

## Installation instructions

1. Create a conda environment : `conda create --name gbm6700e python=3.8`
2. Activate the environment : `conda activate gbm6700e`
3. Install the package : `pip install -e 'git+https://github.com/vtrno/PyGBM6700E.git@main#egg=spinetools&subdirectory=lab_1'`

## Usage

In your python script : `import spinetools`. You can then access the following modules :

* `io` for matlab files reading
* `data` for data processing
* `structures` for data structures management
* `solver` for access to pre-made DLT solver  
* `render` for 3D plotting the vertebrae

## Documentation

Documentation is available in the [wiki](https://github.com/vtrno/PyGBM6700E/wiki) section.