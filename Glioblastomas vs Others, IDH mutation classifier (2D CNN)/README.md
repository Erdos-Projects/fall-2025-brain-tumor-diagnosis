This repository contains models and preprocessing pipelines for MRI-based classification of:

1. Glioblastoma vs Lower-grade Gliomas (GBM vs Others)
2. IDH Mutation Status (IDH-wildtype vs IDH-mutant)

The work combines 2D multi-modal MRI slices with clinical metadata to build a hybrid deep learning framework for precision neuro-oncology.

📂 Notebook: 
Glioblastoma_and_IDH_mutation.ipynb

This notebook includes:
- Preprocessing of the metadata and MRI images
- some EDA needed for the models (sex vs tumor grade, age distribution, IDH mutation, 1p/19q codeletion)
- Glioblastoma vs Others CNN model
    - The dataset consists of 396 patients with Glioblastoma tumors and 99 others. We first set aside 20% of the cases as a held-out test set, and further divided the remaining data into 80% training and 20% validation subsets.
    - For each patient, a single 2D slice corresponding to the plane of maximum tumor area was extracted from the segmentation mask.
    - Modalities: ADC, Flair, T1, T1c, and T2
    - Metadata features fed: sex, age, IDH mutation, 1p/19q codeletion
    - The model uses a 2D CNN with metadata fusion to perform binary classification of glioma grade, distinguishing Glioblastoma from other gliomas.
- IDH mutation classifier CNN model
    - The dataset consists of 392 patients with IDH-wildtype and 103 with IDH-mutated gliomas. We apply the same split as the prior model.
    - For each patient, a single 2D slice corresponding to the plane of maximum tumor area was extracted from the segmentation mask and used as model input across four MRI modalities (FLAIR, T1, T1c, and T2).
    - The model jointly processes imaging and metadata features (sex and age) through a hybrid CNN–fully connected architecture to perform binary classification of IDH status.

GBM_classifier.pth
- This file stores the trained weights for the 2D CNN with metadata fusion used to classify Glioblastoma vs other gliomas. 

IDH_classifier.pth
- Best model for the IDH classifier (the epoch where training loss starts to plateau). This was then used to test against the hold-out test data. 
