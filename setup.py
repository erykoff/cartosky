from setuptools import setup, find_packages

URL = 'https://github.com/kadrlica/cartosky'

with open('requirements.txt') as f:
    install_requires = [req.strip() for req in f.readlines() if req[0] != '#']

setup(
    name='cartosky',
    packages=find_packages(exclude=('tests')),
    package_data={'cartosky': ['data/*.txt', 'data/*.dat']},
    description="Python tools for making sky maps",
    author="Alex Drlica-Wagner, Eli Rykoff, and others",
    author_email='kadrlica@fnal.gov',
    url=URL,
    install_requires=install_requires,
    use_scm_version=True,
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
)
