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

setup(name='pmml',
      version='0.2',
      description='A PMML translator',
      long_description=read_md('README.md'),
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
