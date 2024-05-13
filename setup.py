from setuptools import setup, find_packages
import ep_pred

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name="EP-pred", author="Ruite Xiang", author_email="ruite.xiang@bsc.es",
      description="Prediction of promiscuity in esterases",
      url="https://github.com/etiur/EP-pred", license="MIT",
      version="%s" % ep_pred.__version__,
      packages=find_packages(), python_requires="==3.7", long_description=long_description,
      long_description_content_type="text/markdown",
      classifiers=["Programming Language :: Python :: 3.7",
                   "License :: OSI Approved :: MIT License",
                   "Operating System :: Unix",
                   "Intended Audience :: Science/Research",
                   "Natural Language :: English",
                   "Environment :: Console",
                   "Topic :: Scientific/Engineering :: Bio-Informatics"],
      install_requires=["openpyxl", "scikit-learn", "joblib==1.0.1",
                        "numpy", "pandas", "seaborn", "biopython"],
      keywords="bioprospecting, bioinformatics, machine learning, promiscuity")
