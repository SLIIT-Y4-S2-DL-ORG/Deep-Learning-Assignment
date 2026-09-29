# GRU (Gated Recurrent Unit) Model

This directory contains the GRU implementation for the Human Activity Recognition (HAR) classification task.

## Implementation Details

* **Input:** 128 timesteps × 9 channels (accelerometer and gyroscope)
* **Data Source:** Shared dataset `processed_Data/har_processed.npz`
* **Output:** 6 activity classes
* **Framework:** TensorFlow/Keras
* **Random Seed:** 42
* **Loss Function:** Sparse categorical cross-entropy
* **Optimizer:** Adam
* **Initial Learning Rate:** 0.001
* **Batch Size:** 64
* **Maximum Epochs:** 100
* **Early Stopping:** Validation loss, patience 10
* **Learning Rate Reduction:** Validation loss, factor 0.5, patience 5

## Architecture Candidates

Three GRU architectures were evaluated using the same preprocessed data and experimental methodology as the other deep learning models:

* **GRU_A_Small:** 1 GRU layer (64 units)
* **GRU_B_Medium:** 2 GRU layers (128 → 64 units)
* **GRU_C_Large:** 2 GRU layers (256 → 128 units)

Each GRU layer is followed by Batch Normalization and Dropout (0.30). The final classification layer contains 6 neurons with softmax activation.

## Model Selection

The candidate models were trained using the common training and validation sets. The final GRU architecture was selected using the lowest validation loss, while the test set was kept unseen until final evaluation.

The selected model was:

**GRU_A_Small — 64 GRU units**

### Candidate Results

| Model        | Validation Accuracy | Validation Loss | Parameters | Training Time |
| ------------ | ------------------: | --------------: | ---------: | ------------: |
| GRU_A_Small  |              98.67% |          0.0559 |     15,046 |       75.80 s |
| GRU_B_Medium |              96.95% |          0.0754 |     91,782 |      179.50 s |
| GRU_C_Large  |              96.67% |          0.0830 |    355,590 |      438.54 s |

## Final Test Results

The selected GRU_A_Small model was evaluated once on the unseen test set.

| Metric             |   Result |
| ------------------ | -------: |
| Accuracy           |   90.94% |
| Weighted Precision |   91.62% |
| Weighted Recall    |   90.94% |
| Weighted F1        |   90.79% |
| Macro Precision    |   91.73% |
| Macro Recall       |   90.98% |
| Macro F1           |   90.88% |
| Test Loss          |   0.2945 |
| Parameters         |   15,046 |
| Training Time      |  75.80 s |
| Inference Time     |   1.00 s |

The detailed classification report, standard and normalized confusion matrices, training history, candidate comparison, and final model are available in `Results/GRU/`.

## Results and Artifacts

Results are saved in `../Results/GRU/`:

* `gru_results.csv` — final test metrics
* `gru_candidate_comparison.csv` — comparison of GRU candidates
* `gru_training_history.csv` — training and validation history
* `gru_final.keras` — saved final GRU model

To view the complete implementation, open `GRU.ipynb`.

# Model Performance Comparison

This repository contains the benchmarking results for various deep learning architectures evaluated on our dataset. The goal was to find the optimal balance between classification performance and computational efficiency.

## Overall Results Summary

The table below outlines the performance metrics across four different models: Multi-Layer Perceptron (MLP), 1D Convolutional Neural Network (1D CNN), Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU).

| Model | Test Accuracy | Test F1 | Parameters | Training Time | Key Observation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| MLP | *TBD* | *TBD* | *TBD* | *TBD* | *Pending evaluation* |
| 1D CNN | *TBD* | *TBD* | *TBD* | *TBD* | *Pending evaluation* |
| LSTM | *TBD* | *TBD* | *TBD* | *TBD* | *Pending evaluation* |
| **GRU** | **90.94%** | **90.79%** | **15,046** | **75.80s** | **Small GRU selected as baseline** |