# Human Activity Recognition using Deep Learning

This repository contains the source code and experimental configurations for the SE4050 Deep Learning assignment. The project evaluates and compares four distinct neural network architectures on sequential sensor data to classify human activities.

*   **MLP:** Baseline feedforward network.
*   **1D CNN:** Extracts local temporal patterns from sensor sequences.
*   **LSTM:** Captures long-term temporal dependencies.
*   **GRU:** Evaluates recurrent performance with a simplified gating mechanism.

## Dataset Access

The models are trained and evaluated on the Human Activity Recognition Using Smartphones Data Set from the UCI Machine Learning Repository.

Download the [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) and extract the contents into a `data/raw/` directory at the root of this repository before executing the data preprocessing scripts.

## Environment Setup

The following terminal commands will configure the virtual environment, install the dependencies, and launch the workspace.

```bash
git clone https://github.com/SLIIT-Y4-S2-DL-ORG/Deep-Learning-Assignment.git
cd <your-model-directory>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
code .
```

## Execution Instructions

All preprocessing and hyperparameter tuning decisions are made using only the training and validation data, ensuring the test dataset remains completely unseen for the final evaluation. Saved configurations and random seeds are utilized throughout to maintain reproducibility.

*   Execute `jupyter notebook` from your terminal to launch the interactive environment.
*   Run `preprocessing.ipynb` to format the raw sequential sensor windows and prevent data leakage.
*   Run `train_models.ipynb` to train the selected architectures using the saved random seeds.
*   Run `evaluate.ipynb` to generate the comparison metrics, confusion matrices, and performance visualizations.
