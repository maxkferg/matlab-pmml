from setuptools import setup

setup(name='pmml',
      version='0.2',
      description='A PMML translator',
      url='http://github.com/storborg/pmml',
      author='EIG',
      author_email='pengbo.sherry@gmail.com',
      license='EIG',
      packages=['pmml'],
      install_requires=[
          'lxml',
          'numpy',
          'datetime',
      ],
      zip_safe=False)
