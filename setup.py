from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='fdd',
      version='0.0.1',
      description='Package for FDD',
      url='http://github.com/akionakamura/fdd',
      author='Thiago Akio Nakamura',
      author_email='akionakas@gmail.com',
      license='',
      packages=['fdd'],
      install_requires=[
          'markdown',
          'sklearn',
          'numpy',
          'scipy',
      ],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'])