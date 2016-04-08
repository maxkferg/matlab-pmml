# PMML for Gaussian Process Regression
This repository contains a set of packages for R, Matlab and Python to save
trained Gaussian Process Regression models to PMML. The API is reasonably consistent
across all three languanges, but is adapted to the style conventions of each language.

## PMML for Matlab
This package allows trained GPR models to be saved to PMML and loaded again.
Internally this package uses the GPML package for scoring and optimization
of hyperparameters. See the Matlab folder for full documentation.

## PMML for Python
This package allows trained GPR models to be save to PMML and loaded again.
Internally this package uses the SciKit package for scoring and optimization
of hyperparameters.

## PMML for R
(Yet to be implimented)

## Todo
- Support 'AbsoluteExponentialKernel' 'GeneralizedExponentialKernel'
- Support for more than 2 feature dimensions in Matlab package
- Impliment R package
- Test each package with a large number of inputs
- Support column naming (other than 'x1','x2',...)

## License
MIT