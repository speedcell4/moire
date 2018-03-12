from setuptools import setup

setup(
    name='moire',
    version='0.0.1',
    packages=[
        'launch_moire',
        'moire',
        'moire.nn',
        'moire.nn.connections',
        'moire.nn.convolutions',
        'moire.nn.recurrents',
        'moire.nn.reinforces',
        'moire.nn.sparses',
    ],
    url='',
    license='MIT',
    author='speedcell4',
    author_email='speedcell4@gmail.com',
    description='DyNet interfaces for Humans',
)
