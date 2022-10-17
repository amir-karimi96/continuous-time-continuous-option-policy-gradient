from setuptools import setup, find_packages


setup(name='CTCO',
      packages=[package for package in find_packages()
                if package.startswith('CTCO')],
      description='Continuous-time Continuous-option Reinforcement Learning for continuous control',
      author='CTCO Team',
      url='https://github.com/amir-karimi96/continuous-time-continuous-option-policy-gradient.git',
      author_email='amirmoha@ualberta.ca',
      version='1.0.0')