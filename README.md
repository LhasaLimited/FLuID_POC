# FLuID_POC
Code to support the FLuID Proof-of-Concept publication.

## System Requirements
### Hardware requirements
The FLuID Proof of Concept requires only a standard computer with enough RAM to support the in-memory operations.

### Software requirements
#### OS Requirements
FLuID has been run on Linux and Windows 10, but it should run in any python environment that is capable of running a Jupyter Notebook.
It is possible to use a GPU to speed up the learning process, but it isn't mandatory.

Windows: 10, and 11
Linux: Centos

#### Python Dependencies
The FluID dependencies are found in the yml file.

```
  - rdkit
  - tmap
  - scikit-learn
  - imbalanced-learn
  - xgboost
  - tqdm
  - plotly
  - ipywidgets
  - faerun
  - mhfp
  - seaborn
```

## Installation Guide:

### Install from Github
```
git clone https://github.com/LhasaLimited/FLuID_POC
```
Switch to the release branch FLuID_paper.

Install the requirements from the .yml file.

Typical Install time is less than 5 minutes, assuming you have a working python environment.

## Running
Open the Jupyter notebook and then execute. A seed has been set to control for reproducability, but the results will not be completely identical due to uncontrollable randomness in learning algorithms.
Typical runtime on a normal computer is approximately half an hour.
