import numpy as np

data = np.load("../processed_Data/har_processed.npz", allow_pickle=True)

train_subjects = set(data["subjects_train"])
val_subjects = set(data["subjects_val"])
test_subjects = set(data["subjects_test"])

print("Train subjects:", sorted(train_subjects))
print("Val subjects:", sorted(val_subjects))
print("Test subjects:", sorted(test_subjects))

print("Train ∩ Val:", train_subjects & val_subjects)
print("Train ∩ Test:", train_subjects & test_subjects)
print("Val ∩ Test:", val_subjects & test_subjects)
print("Val ∩ Test:", val_subjects & test_subjects)