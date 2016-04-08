"""
Test the simple example described in the PMML documentation
This file is used to ensure that the GaussianProcessModel is working
exactly as is shoulc according to the NIST documentation
"""
import sys
import logging
import unittest
import numpy as np
from ..gaussian_process import GaussianProcessModel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

class TestGaussianProcess(unittest.TestCase):

    def setup(self):
        self.expected = 'expected.pmml'
        self.output = 'fixtures/output.pmml'


    def teardown(self):
    	pass


    def test_loadFromFile(self):
    	"""Test that the trained object can be read from a valid PMML file"""
        self.expected = 'pmml/tests/fixtures/expected.pmml'
        self.output = 'pmml/tests/fixtures/output.pmml'
    	gpm = GaussianProcessModel(self.expected)
        assert(gpm.kernelName=='ARDSquaredExponentialKernelType')
    	assert(gpm.nugget==0.011046)
        assert(gpm.xTrain.size==4)
    	assert(gpm.yTrain.size==2)
    	assert(gpm.gamma==2.489)


    def testLoadFromGPObject(self):
        """Test that the GaussianProcessModel can be loaded from a trained GP object"""
        xTrain = np.array([[1,3],[2,6]])
        yTrain = np.array([[1],[2]])
        xNew = np.array([[1,4]])
        # Train a GP model
        hyp = [1,50]
        s = 0.1;
        k = 1.0 * RBF(hyp) + WhiteKernel(noise_level=s**2, noise_level_bounds=(1e-3,1))
        gp = GaussianProcessRegressor(kernel=k,n_restarts_optimizer=0)
        gp.fit(xTrain, yTrain)
        # Create a new GaussianProcessModel
        gpm = GaussianProcessModel(gp);
        # gpm.toPMML(self.output)
        # Test that the scoring works correctly
        [mu,s] = gpm.score(xNew);
        self.assertAlmostEqual(mu, 1.0095, places=3)
        self.assertAlmostEqual(s**2, 0.0226,  places=1)


    def test_loadFromFileAndScore(self):
        """
        Test that the trained object can be read from a valid PMML file
        and used to score some new x values
        """
        self.expected = 'pmml/tests/fixtures/expected.pmml'
        gpm = GaussianProcessModel(self.expected)
        xNew = np.atleast_2d([1,4]);
        [mu,s] = gpm.score(xNew);
        self.assertAlmostEqual(mu, 1.0095, places=4)
        self.assertAlmostEqual(s**2, 0.0226,  places=4)


    def test_createPMML(self):
        """Test that we can create valid PMML from a GaussianProcessModel object"""
        xTrain = np.array([[1,3],[2,6]])
        yTrain = np.array([[1],[2]])
        xNew = np.atleast_2d([1,4])
        # Train a GP model
        hyp = [1,59]
        s = 0.1;
        k = 1.0 * RBF(hyp) + WhiteKernel(noise_level=s**2, noise_level_bounds=(1e-3,1))
        gp = GaussianProcessRegressor(kernel=k,n_restarts_optimizer=0)
        gp.fit(xTrain, yTrain)
        # Create a new GaussianProcessModel
        gpm = GaussianProcessModel(gp);
        # Save to PMML
        gpm.toPMML('pmml/tests/fixtures/output.pmml')


    def text_createPMMLandPredict(self):
        """Test that we can create valid PMML and then use it to predict"""
        filename = 'pmml/tests/fixtures/output.pmml'
        xTrain = np.array([[1,3],[2,6]])
        yTrain = np.array([[1],[2]])
        xNew = atleast_2d([1,4])
        # Train a GP model and save to PMML
        hyp = [1,59]
        s = 0.1;
        k = 1.0 * RBF(hyp) + WhiteKernel(noise_level=s**2, noise_level_bounds=(1e-3,1))
        gp = GaussianProcessRegressor(kernel=k,n_restarts_optimizer=0)
        gp.fit(xTrain, yTrain)
        gpm = GaussianProcessModel(gp);
        gpm.toPMML(filename)

        # Load from pmml and predict
        gpm = GaussianProcessModel(filename)
        [mu,s] = gpm.score(xNew);
        self.assertAlmostEqual(mu, 1.0095, places=4)
        self.assertAlmostEqual(s**2, 0.0226,  places=4)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr)
    logging.getLogger("TestNistExample").setLevel(logging.DEBUG)



