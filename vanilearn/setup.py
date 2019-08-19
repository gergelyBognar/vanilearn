from setuptools import setup, find_packages


setup(
    name='vanilearn',
    version='0.0.1',
    description='An implementation of decision tree and random forest algorithms.',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    install_requires=[
        'numpy',
    ],
)
