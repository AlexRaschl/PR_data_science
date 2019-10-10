import setuptools

VERSION = '0.1'

setuptools.setup(
    name='pr_data_science_project',
    version=VERSION,
    author='Alexander R',
    description='PR Data Science Project',
    packages=[
        'src',
        'src/database',
        'src/extract',
    ],
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent'
    ],
    install_requires=[
        'matplotlib',
        'numpy',
        'tensorflow',

    ]
)
