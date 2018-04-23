from setuptools import setup

setup(
    name='moire',
    version='0.0.1',
    packages=[
        'launch_moire',
        'moire',
        'moire.nn',
        'moire.nn.attentions',
        'moire.nn.connections',
        'moire.nn.convolutions',
        'moire.nn.normalizations',
        'moire.nn.recurrents',
        'moire.nn.reinforces',
        'moire.nn.reinforces.agents',
        'moire.nn.scheduling',
        'moire.nn.sparses',
        'moire.nn.functions',
    ],
    url='',
    license='MIT',
    author='speedcell4',
    author_email='speedcell4@gmail.com',
    description='DyNet interfaces for Humans',
)
