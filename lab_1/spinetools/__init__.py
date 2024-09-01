"""
SpineTools was built for the course GBM6700E at Polytechnique Montréal. It's meant to take over Matlab, which was used before, and provide easy access to Python tools for reading and processing the data. 

Modules :  
    - io : Handle inputs and outputs  
    - data : Process data structures  
    - render : Pretty 3D plots  
    - solver : Wrapper for the DLT (mainly)  
    - structures : Data structures for easy viz and management  
"""

from . import io
from . import data
from . import render
from . import solver
from . import structures