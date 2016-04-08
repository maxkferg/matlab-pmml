from setuptools import setup

# PyPi only supports rst but Github requires markdown.
# Documentation is written in markdown and converted to rst on build
# Pypandoc should be installed by the user who is builing this package
# See http://stackoverflow.com/questions/10718767/have-the-same-readme-both-in-markdown-and-restructuredtext
try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()

packages = [
  'lxml',
  'numpy',
  'datetime',
],

setup(name='pmml',
      version='0.3',
      description='A PMML translator',
      long_description=read_md('README.md'),
      url='https://github.com/maxkferg/pmml-gpr/tree/master/matlab',
      author='EIG',
      test_suite='tests',
      author_email='maxkferg@stanford.edu',
      license='EIG',
      packages=['pmml'],
      install_requires=packages,
      tests_require=packages,
      zip_safe=False)
