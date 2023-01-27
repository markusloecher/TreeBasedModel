# Augmented Hierarchical Shrinkage

A python package used to develop and test an augmented version of the Hierarchical Shrinkage regularization method for random forests. In contrast to the original HS version developed by [1] we introduce an additional penalty for the node-wise degree of overfitting to minimize the influence of splits caused by the biased feature selection process in tree-based models.

We developed two methods of AugHS that use the readily available OOB samples in RF to estimate the model’s degree of overfitting: **AugHS smSHAP** and **AugHS MSE**.

**AugHS smSHAP** uses an unbiased feature importance measure called smooth SHAP [2] to estimate the node-wise degree of overfitting on a feature level. Whenever an uninformative split occurs, the respective split contribution is penalized, regardless of its position in the tree.

In contrast, **AugHS MSE** uses OOB data to recalculate the node values of the trees. Subsequently, we use a hypothesis testing framework to assess how much the newly calculated node values differ from their in-bag versions. If a node shows bias, we reduce its effect on the predictive outcome.

Both versions of AugHS are enclosed in the self-developed python module `TreeModelsfromScratch`, which features algorithms for decision trees and random forests for classification and regression tasks, and tools for model selection, model evaluation, and other utilities. The code for the "standard" Decision Tree and Random Forest algorithm is taken from the [MLfromscratch repository](https://github.com/patrickloeber/MLfromscratch).

## Installation

### Clone repository
1. Open the Terminal.
2. Change the current working directory to the location where you want the cloned directory
3. Clone the repository using
```bash
git clone git@github.com:Heity94/AugmentedHierarchicalShrinkage.git
```
4. Change into the cloned directory
```bash
cd AugmentedHierarchicalShrinkage/
```
### Create and activate virtual environment
To avoid dependency issues we recommend to create a new virtual environment to install and run our code. Please note, that in order to follow the next steps [pyenv](https://github.com/pyenv/pyenv) and [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) need to be installed.
1. Create a virtual environment using `pyenv virtualenv`. You can of course use another name as `aug_hs_env` for the environment.
```bash
pyenv virtualenv aug_hs_env
```
2. Optionally: Use `pyenv local` within the AugmentedHierarchicalShrinkage source directory to select that environment automatically:
```bash
pyenv local aug_hs_env
```
From now on, all the commands that you run in this directory (or in any sub directory) will be using `aug_hs_env` virtual env.

### Install the package

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install AugmentedHierarchicalShrinkage.
```bash
pip install --upgrade pip
pip install -e .
```

## Usage

After successful installation you should be able to use the `TreeModelsfromScratch` module to instantiate Random Forest models with AugHS and run and recreate the experiments in the `notebooks` folder.

A detailed explanation on how to use the module to instantiate and train different RF models can be found in [`notebooks/Documentation_TreeModelsfromScratch.ipynb`](/notebooks/Documentation_TreeModelsfromScratch.ipynb)

```python
# Imports
from TreeModelsFromScratch.RandomForest import RandomForest
from TreeModelsFromScratch.datasets import DATASETS_CLASSIFICATION, DATASET_PATH
from sklearn.model_selection import train_test_split
from imodels import get_clean_dataset

# prepare data (a sample "haberman" dataset)
dset_name, dset_file, data_source = DATASETS_CLASSIFICATION[3]
X, y, feat_names = get_clean_dataset(dset_file, data_source, DATASET_PATH)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Instantiate and train regular RF for classification (without additional regularization)
rf = RandomForest(n_trees=25, treetype='classification')
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Instantiate and train RF with HS applied post-hoc
rf_hs = RandomForest(n_trees=25, treetype='classification', HShrinkage=True, HS_lambda=10)
rf_hs.fit(X_train, y_train)
y_pred = rf_hs.predict(X_test)

# Instantiate and train RF with AugHS smSHAP applied post-hoc
rf_aug_smSH = RandomForest(n_trees=25, treetype='classification', HS_lambda=10, oob_SHAP=True, HS_smSHAP=True)
rf_aug_smSH.fit(X_train, y_train)
y_pred = rf_aug_smSH.predict(X_test)

# Instantiate and train RF with AugHS MSE applied post-hoc
rf_aug_mse = RandomForest(n_trees=25, treetype='classification', HS_lambda=10, HS_nodewise_shrink_type="MSE_ratio")
rf_aug_mse.fit(X_train, y_train)
y_pred = rf_aug_mse.predict(X_test)
```

## Structure of the repository
The general structure of this repository is detailed below
```bash
.
├── TreeModelsFromScratch   # Python module for decision tree and random forest models
├── scripts                 # Script to run predictive performance experiment
├── raw_data                # Datasets used for training and evaluating the artifact
├── notebooks               # Notebooks to run experiments, recreate original results from HS paper and compare self-developed models with sklearn and imodels implementation
├── data                    # Experimental results, including trained models, simulation settings, and created plots
├── MANIFEST.in
├── README.md
├── .gitignore
├── requirements.txt
└── setup.py
```

## References
[1] Abhineet, Agarwal; Tan, Yan Shuo; Ronen, Omer; Singh, Chandan; Yu, Bin (2022): Hierarchical Shrinkage: Improving the accuracy and interpretability of tree-based models. In International Conference on Machine Learning, pp. 111–135. Available online at [https://proceedings.mlr.press/v162/agarwal22b.html](https://proceedings.mlr.press/v162/agarwal22b.html).

[2] Loecher, Markus (2022): Debiasing MDI Feature Importance and SHAP Values in Tree Ensembles. In Andreas Holzinger, Peter Kieseberg, A. Min Tjoa, Edgar Weippl (Eds.): Machine Learning and Knowledge Extraction, vol. 13480. Cham: Springer International Publishing (Lecture Notes in Computer Science), pp. 114–129.
