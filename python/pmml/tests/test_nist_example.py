"""
Test the simple example described in the PMML documentation
This test is used to make sure that the SCIKIT gaussian_process function
behaves as we would suspect.
"""
import sys
import pmml
import logging
import unittest
import numpy as np
import matplotlib
from sklearn import gaussian_process
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


log = logging.getLogger("TestNistExample")

class TestNistExample(unittest.TestCase):
    def setup(self):
        pass

    def teardown(self):
        pass

    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test_kernels(self):
        """
        Test that the RBF kernel matches the squared_exponential kernel from
        the nist documentation.
        """
        xTrain = np.array([[1,3],[2,6]])
        yTrain = np.array([[1],[2]])
        xTest = np.array([[1,4]])
        xWild = np.array([[1,9]])
        hyp = [1,59]
        s = 0.1;
        k = 1.0 * RBF(hyp) + WhiteKernel(noise_level=s**2, noise_level_bounds=(1e-3,1))

        gp = GaussianProcessRegressor(kernel=k,n_restarts_optimizer=0)
        gp.fit(xTrain, yTrain)

        # The optimized kernel can now be accessed at gp.kernel_
        log.info("Now we have a trained kernel!!!")
        log.info(gp.kernel_)
        k = gp.kernel_
        r = k(xTest,xWild)
        log.info("New RBF Kernel gives value %f"%r)

        [mu,s] = gp.predict(xTest,return_std=True)
        log.info("GPR with RBF Kernel gives value mu=%f and var=%f"%(mu,s**2))


    def test_optimizer(self):
        """Plot the LML landscape to test the GP optimizer"""
        # Setup the model
        xTrain = np.array([[1,3],[2,6]])
        yTrain = np.array([[1],[2]])
        xTest = np.array([[1,4]])
        xWild = np.array([[1,9]])
        hyp = [1,59]
        s = 0.1;
        k = 1.0 * RBF(hyp) + WhiteKernel(noise_level=s**2, noise_level_bounds=(1e-3,1))

        gp = GaussianProcessRegressor(kernel=k,n_restarts_optimizer=0)
        gp.fit(xTrain, yTrain)

        # Plot LML landscape
        plt.figure(1)
        theta0 = np.logspace(0, 1, 49)
        theta1 = np.logspace(-4, 1, 50)
        Theta0, Theta1 = np.meshgrid(theta0, theta1)
        LML = [[gp.log_marginal_likelihood(
            np.log([Theta0[i, j], 1.52 ,50,Theta1[i, j]])
        ) for i in range(Theta0.shape[0])] for j in range(Theta0.shape[1])]
        LML = np.array(LML).T

        vmin, vmax = (-LML).min(), (-LML).max()
        vmax = 100
        plt.contour(Theta0, Theta1, -LML,
                    levels=np.logspace(np.log10(vmin), np.log10(vmax), 100),
                    norm=LogNorm(vmin=vmin, vmax=vmax))
        plt.colorbar()
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Length-scale")
        plt.ylabel("Noise-level")
        plt.title("Log-marginal-likelihood")
        plt.tight_layout()
        plt.show()



def testHyparameters(hyp):
    """Test the hyperparameters against the nist example. Throw on failure"""
    pass


def testPrediction(ynew):
    """Test the predicted value against the nist example. Throw on failure"""
    pass



if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr)
    logging.getLogger("TestNistExample").setLevel(logging.DEBUG)
    unittest.main()
    TestNistExample().test_gpr()




