import os

import pkg_resources
from setuptools import setup, find_packages
 
setup(
    name="Effdet",
    # py_modules=["Effdet"],
    version="1.0",
    description="EfficientDET Modified in Pytorch",
    author="zylo117",
    packages=find_packages(exclude=['logs','projects','datasets','res'],include=['*']),
    # py_modules=["effdet"],
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
    classifiers=[
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
        ],
    extras_require={'dev': ['pytest']},
)