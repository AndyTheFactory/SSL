from setuptools import find_packages
from setuptools import setup


setup(name='FakeNewsKeras',
      version='1.0.1',
      description='Fake News Classifier',
      author='Andrei Paraschiv',
      author_email='a@nup.ro',
      url='https://github.com/AndyTheFactory/C/FakeNewsKeras',
      license='MIT',
      install_requires=['Keras==2.1.2'],
      packages=find_packages())
