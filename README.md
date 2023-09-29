# Characterisation on the oncogenic effect of the missense mutations of p53 via machine learning

## Source code for Reproduction

### Note

These codes are supposed to run on a Linux machine.

Here is a short introduction about the statistical test and machine learning analysis of the p53 mutation analysis.

### Contents

This repository contains:

1. Original dataset from the paper [TP53_PROF](https://academic.oup.com/bib/article/23/2/bbab524/6510957)
2. All 289 features generated in our paper before feature selection
3. The final datasets of the *ExpAssay* and *noExpAssay* models.


### Statistical test

All the codes to compute the p-value between functional and non-functional mutations are in `run_statistics.R`. 

Please use R program to run this code in the terminal or Rstudio.

```bash
# running in the terminal
Rscript run_statistics.R
```

### Machine learning analysis

#### 1. Setting up the environment

We suggest to use `conda` for environment setting. Please go to the website for more information about the installation, such as https://docs.conda.io/projects/miniconda/en/latest/

After you get `conda` installed, simply use the `requirements.yml` file to set up the environemnt

```bash
# set up conda env
conda env create -f requirements.yml

# activate env
conda activate p53_analysis
```

#### 2. Train and test our models

You can run the `run_machine_learning.py` script to check the results of our *ExpAssay* and *noExpAssay* model on the performance of 10-fold Cross Validation and Blind test

```bash
# train and test the performance
python run_machine_learning.py > result.txt
```

## Contact

If you have any questions, please visit our website 

https://biosig.lab.uq.edu.au/




