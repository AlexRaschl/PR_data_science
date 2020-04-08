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
        'src/preprocessing',
        'src/model',
        'src/protocol',
        'src/scraping',
        'src/download'
    ],
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent'
    ],
    install_requires=[
        'matplotlib',
        'numpy',
        'tensorflow',
        'google-api-python-client',
        'scikit-image',
        'scrapy',
        'seaborn',
        'cv2',
        'youtube_dl'
    ]
)
