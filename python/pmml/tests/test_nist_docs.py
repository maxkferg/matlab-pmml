"""
Test the simple example described in the PMML documentation
"""
import pmml
import unittest
from sklearn import gaussian_process

class TestSimpleExample:
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

    def test_translator(self):
		print "Testing translator:"
		X_train = [[1,3],[2,6]]
		Y_train = [1,2]
		X_test=[[1,4]]
		nugget = 0.01

		# fit GP model
		gp = gaussian_process.GaussianProcess(theta0=[0.1,0.1],nugget=nugget)
		gp.fit(X_train, Y_train)
		p = pmml.GP()

		# write gp model into a PMML file
		print "Writing xml file"
		p.GP_translator('test.xml',gp)



