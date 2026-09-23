# Dev: Dewmini

## LSTM – Human Activity Recognition

This notebook trains three LSTM candidate architectures on the preprocessed
UCI HAR dataset, selects the best one by validation loss, and reports final
test-set metrics using the same evaluation protocol as every other model in
the project.

### Candidate architectures

| Name | LSTM layers |
|------|-------------|
| LSTM_A_Small | 64 |
| LSTM_B_Medium | 128 → 64 |
| LSTM_C_Large | 256 → 128 |

Each LSTM layer is followed by BatchNormalization + Dropout(0.3).
The final Dense layer has 6 softmax outputs (one per activity class).

### How to run

1. Make sure `processed_Data/har_processed.npz` exists (run the
   preprocessing notebook first).
2. Open `LSTM/LSTM.ipynb` and **Run All Cells**.
