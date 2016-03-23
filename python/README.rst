
This package will only take Gaussian process models trained by sklearn package.

pmml package contains three functions so far:
1. GP_translator(file_name,model):
	file_name: the PMML file will saved in file_name
	model: a trained GP model
	This function will not return anything

2. GP_parser(file_name):
	file_name: the name of PMML file that will be read and parsed.
	This function will return all the GP parameters, such as lambda, noise term…

3.GP_score(test_data):
	test_data: test data
	This function will return the scoring results.

So far, it is a preliminary package with limitation of:
1. It will not show the error when users use it in a wrong way, for example: user passed a wrong model.

2. As sklearn package will only operate on numerical matrix, which is pre-processed matrix only contains number, there are some function in PMML will not be needed, such as .
	The GP_translator is able to automatically generate PMML file from a Gaussian process model, which contains: a Header, Datadictionary, Gaussian Process Model, Mining schema, output, LocalTransformations, kernel type, and GaussianProcessDictionary.

3. As there is only one type of output, the output form is fixed.

4. As the sklearn package for Gaussian Process only uses normalization, only the normalized transformation is included in this translator.

————————————
example:
————————————
import pmml
from sklearn import gaussian_process


X_train=[[1,1],[2,2],[4,4],[7,7]]
Y_train=[1,2,4,6]
X_test=[[1,3],[2,3]]
nugget=0.01

"""fit GP model"""
gp = gaussian_process.GaussianProcess(theta0=[0.1,0.1],nugget=nugget)
gp.fit(X_train, Y_train)

"""test"""
p=pmml.GP()
# write gp model into a PMML file
p.GP_translator(‘sample.xml',gp)
# read and parse the PMML file
p.GP_parser(‘sample.xml')
# Score test data
prediction=p.GP_score(X_test)

print prediction