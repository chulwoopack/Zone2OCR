#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='zone2ocr',
      version='0.1.0',
      license='GPL',
      url='https://github.com/chulwoopack/Zone2OCR',
      description='Segmentation mapping tool for document processing',
      packages=find_packages(exclude=['exps*']),
      install_requires=[
        'numpy==1.16.2',
        'opencv-python',
        'shapely',
        'tqdm==4.31.1',
        'jupyter',
        'matplotlib'
      ],
      zip_safe=False)
