# Gaussian Process Regression PMML Support for Python

Save and load a trained Gaussian Process regression (GPR) model to/from PMML. This package exposes the
`pmml.gausian_process.GaussianProcessModel` class which is used to represent a trained GPR model. The model hyperparameters
can be optimized using the SciKit package, or directly on any `GaussianProcessModel` object (TODO).
`GaussianProcessModel` objects can be used to generate scores for new x values, regardless of whether they
were initialized from a PMML file, or a trained GPML model.

## Creating GaussianProcessModel objects

### GaussianProcessModel(<GaussianProcessRegressor>)
Create a new GaussianProcess object from an trained Scikit GPR Model

Where:
* <GaussianProcessRegressor> is a trained GaussianProcessRegressor object from the sklearn.gaussian_process package.

Right now we assume that the GaussianProcessRegressor uses a ARDSquaredExponentialKernel. This needs to be tidied up at some stage; We probably need to write separate classes to represent each of the four allowable Kernals [RadialBasisKernel,ARDSquaredExponentialKernel,AbsoluteExponentialKernel,GeneralizedExponentialKernel]

### GaussianProcessModel(filename)
Create a new GaussianProcessModel object from an existing PMML file.
This method of creating GaussianProcess objects is used to load trained models from PMML.

Where:
* filename - the path to a valid PMML filename

## Object methods
Once a GaussianProcess object has been created it can be used to score new
x values or it can be saved to a PMML file. For this section, assume that
`p` is a valid GaussianProcess object.

### p.score(xNew)
Return scores for the new x values. xNew should be an m x n matrix of values
where each row represents a test point. The method will return an m x 1
column vector of y values (scores).

### p.toPMML(filename)
Return the trained GPR model as valid PMML. If the optional filename
parameter is provided, the PMML will be saved to file.

## Example

```python
    import sys
    import numpy as np
    from pmml.gaussian_process import GaussianProcessModel
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    # Define valid function inputs matching the NIST documentation example
    # The hyperparameters are defined in the same way that gpml returns them
    # This make the PMML package easier to use with gpml, but requires the
    # PMML package to make conversions internally
    filename = 'output.pmml'
    xTrain = np.array([[1,3],[2,6]])
    yTrain = np.array([[1],[2]])
    xNew = atleast_2d([1,4])

    # Train a GP model and save to PMML
    hyp = [1,59]
    s = 0.1;
    k = 1.0 * RBF(hyp) + WhiteKernel(noise_level=s**2, noise_level_bounds=(1e-3,1))
    gp = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=0)
    gp.fit(xTrain, yTrain)
    p = GaussianProcessModel(gp);

    # Score some values
    [mu,s] = p.score(xNew);
    self.assertAlmostEqual(mu, 1.0095, places=4)
    self.assertAlmostEqual(s**2, 0.0226,  places=4)

    # Save the model to PMML
    p.toPMML(filename)
```
The GPR model and training points are now saved in the PMML format.
The model can be loaded and used to score some new values.

```python
    # Load from pmml and predict
    xNew = np.atleast_2d([1,4])
    p = GaussianProcessModel(filename)
    [mu,s] = p.score(xNew);
    self.assertAlmostEqual(mu, 1.0095, places=4)
    self.assertAlmostEqual(s**2, 0.0226,  places=4)

    # Score multiple trianing points
    xPoints = np.array([1,4],[2,3],[4,5])
    p.score(xPoints)
```

## Requirements
This package uses the relatively new GaussianProcessRegressor class from scikit. This class is available in the [0.18.dev0 version](https://github.com/scikit-learn/scikit-learn) of scikit.

## TODO
- Support for 'RadialBasisKernel' and 'GeneralizedExponentialKernel' in Matlab package
- Test with a large number of inputs
- Support column naming (other than 'x1','x2',...)

## License
MIT

