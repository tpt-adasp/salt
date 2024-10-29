from setuptools import setup, find_packages

setup(
    name='py-salt',
    version='0.0.1',
    author='Paraskevas Stamatiadis',
    python_requires='>=3.10, <3.12',
    install_requires=[
        'pandas>=2.1.0',
        'numpy>=1.26, <2.0'
        'tqdm>=4.65.0',
        'matplotlib>=2.2.3',
        'networkx>=3.1'],
    extras_require={
        'examples': ['python-Levenshtein>=0.12.0'],
        'tests': ['pytest>=6.2.5'],
        'notebooks': ['jupyter>=1.0.0',
        'ipykernel>=6.26.0',
        'notebook>=6.5.4']},
    packages=find_packages(include=['pysalt*'])
)
