
This library gathers a set of functions that ease processing of the CMIP6 climate projections from the Climate Data Store.

In addition to downloading and processing the data, the library provides a set of functions to plot the data in a consistent way.
It also provides a function that calculate the joint distribution of change in precipitation and temperature and randomly sample from this distribution.

The library will be available from PyPI in the near future, however, you can install it locally by cloning this Github repository and install it with `pip`.

To install the library locally:
  - clone this Github repo on your local machine.
  - from the `CST_PlottingTools` directory, run the command: `python setup.py bdist_wheel sdist`.
  - next, once your conda environment activated, from the `CST_PlottingTools` directory, run the command `pip install .`
