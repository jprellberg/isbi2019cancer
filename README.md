# C-NMC Challenge

This is the code release for the paper:

Prellberg J., Kramer O. (2019) Acute Lymphoblastic Leukemia Classification from Microscopic Images Using Convolutional Neural Networks. In: Gupta A., Gupta R. (eds) ISBI 2019 C-NMC Challenge: Classification in Cancer Cell Imaging. Lecture Notes in Bioengineering. Springer, Singapore

## Usage

Use the script `main_manual.py` to train the model on the dataset. The expected training data layout is described below.

Use the script `submission.py` to apply the trained model to the test data.

## Data Layout

The training data during the challenge was released in multiple steps which is why the data layout is a little peculiar.

```
data/fold_0/all/*.bmp
data/fold_0/hem/*.bmp
data/fold_1/...
data/fold_2/...
data/phase2/*.bmp
data/phase3/*.bmp
data/phase2.csv
```

The `fold_0` to `fold_2` folders contain the training images with two subdirectories for the two classes each. The directories `phase2` and `phase3` are the preliminary test-set and test-set respectively and contain images numbered starting from `1.bmp`. The labels for the preliminary test-set are specified in `phase2.csv` which looks as follows:

```
Patient_ID,new_names,labels
UID_57_29_1_all.bmp,1.bmp,1
UID_57_22_2_all.bmp,2.bmp,1
UID_57_31_3_all.bmp,3.bmp,1
UID_H49_35_1_hem.bmp,4.bmp,0
```
