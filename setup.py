from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='HNSC-classifier',
    version='1.0',
    packages=find_packages(),
    entry_points={
        "console_scripts": ['HNSC-classifier = main:run',]
    },
    url='https://github.com/yangfangs/HNSC-classifier',
    license='GNU General Public License v3.0',
    author='Yang Fang',
    author_email='506528950@qq.com',
    description='HNSC classifier',
    install_requires=required,
    package_data={'': ['*.md',"*txt"]},
)
