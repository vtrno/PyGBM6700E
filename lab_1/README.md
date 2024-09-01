# SpineTools

Python package for GBM6700E course lab 1 at Polytechnique Montréal

## Installation instructions

1. Create a conda environment : `conda create --name gbm6700e python=3.8`
2. Activate the environment : `conda activate gbm6700e`
3. Clone this repo : `git clone https://github.com/vtrno/PyGBM6700E`
4. Navigate to this folder : `cd PyGBM6700E/lab_1`
5. Build the whl package : `python -m pip wheel --no-deps .`
6. Install the package : `python -m pip install spinetools-1.0.0-py3-none-any.whl`

## Usage

In your python script : `import spinetools`. You can then access the following modules :

* `io` for matlab files reading
* `data` for data processing
* `structures` for data structures management
* `solver` for access to pre-made DLT solver  
* `render` for 3D plotting the vertebrae

## Documentation

Documentation is available in the `docs`folder. Just open the [index.html](docs/index.html) file in your favorite web browser !