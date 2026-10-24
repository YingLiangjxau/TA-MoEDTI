# TA-MoEDTI

## System Requirements

The source code developed in Python 3.8 using PyTorch 2.4.0. The required python dependencies are given below. DrugBAN is supported for any standard computer and operating system (Windows/macOS/Linux) with enough RAM to run. There is no additional non-standard hardware requirements.

```xml
codna install torch == 2.4.0
pip install torch-geometric == 2.6.1
pip intsall jupyter
pip install numpy == 1.24.3
pip install pandas == 2.0.3
pip install tqdm == 4.66.5
pip install scikit-learn == 1.3.0
pip install rdkit == 2024.3.5
pip install matplotlib == 3.7.2
pip install seabor seaborn == 0.13.2

```
## Dataset

The dataset includes eight datasets: 'DrugBank', 'BIOSNAP', 'BindingDB'.

## Preparation
- Data preprocessing: `data_preprocessing.ipynb`
- Heterogeneous graph construction: `HeteroData_preprocessing.ipynb`

## run model
- Experiments: Run `main.py` to execute the experiments.

## Testing generalization capabilities of  TA-MoEDTI
- Cross-dataset generalization experiments: First, run `remove_repeated_edges.py` to remove duplicate edges, then run `cross_mian.py` to perform the generalization experiments.
