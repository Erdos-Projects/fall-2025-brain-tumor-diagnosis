This file directory includes a 2D ResNet18 model using the pytorch and torchvision libraries. The structure is broken down as follows:

ProcessImagesFromNIFTI.ipynb:
- Processing routine to produce images from random slices of the MRI nifti files obtained through the Cancer Imaging Archive (UCSF PDGM)

Calabrese, E., Villanueva-Meyer, J., Rudie, J., Rauschecker, A., Baid, U., Bakas, S., Cha, S., Mongan, J., Hess, C. (2022). The University of California San Francisco Preoperative Diffuse Glioma MRI (UCSF-PDGM) (Version 5) [dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/tcia.bdgf-8v37

- 4 slices are sampled in both the tumor-containing and non-tumor-containing regions of the brain, if possible, along each of the 3 major axes.
- A minimum threshold for the area of the cross-section in a slice, relative to the maximum cross-section along the relevant major axis, is set to 0.3 to guarantee at least a minimum size cross-section in each image. 
- Image files are saved as grayscale 8-bit pixel PNG files. 
- Image labels (tumor/no-tumor) are determined from tumor segmentation in the original dataset. Images are saved in tumor and notumor directories for ease of viewing and label recovery.
- Images are further saved into directories labeled by the modality of the measurement (T1, T1c, T2, FLAIR).
- Image filenames contain the patient_id, longitudinal axis of the MRI cross-section (mri-axis), and the coordinate that the slice is taken from (slice).

MRIDataset.py
- Dataset needs to be initialized with a pandas DataFrame containing patient id, path to MRI image, and a target label (1 for tumor, 0 for no-tumor).
- SampleMapper is a helper function that will construct a DataFrame from a modality and base directory containing the images with the directory structure defined in ProcessImagesFromNIFTI.ipynb. 
- SampleMapper will assign targets based on the directory the image is found in and will only populate the DataFrame with images from the specified modality directory.
- SampleMapper will assign patient_id based on the file name structure. It is important that the convention defined in ProcessImagesFromNIFTI.ipynb is not altered.
- MRIDataset class accepts a DataFrame, matching the structure from SampleMapper, and any potential transformations applied to the data for use in the model.
- Default transformations and flags in the dataset are designed to match the required inputs for the resnet18 model in torchvision, which expects a 3-channel (rgb) image with resolution 128x128.

ResNetModeling.ipynb:
- A full patient list is constructed from the data directory containing all of the patients and their full set of MRI data. Each entry is a patient_id (UCSF-PDGM-XXXX)
- This list can be pulled instead from the preprocessed UCSF-PDGM-metadata_v5.csv. 
- Patients tumor grade propagated into a list. Indicies of this list correspond to the indicies in the full patient list.
- The model is constructed using FLAIR MRI images.
- Training (80%) and test (20%) sets are split and stratified by tumor grade. 
- Training set then split into 4-folds for cross-validation of the model. 
- 4-folds were chosen to maintain somewhat reasonable training time to allow for model iteration and some level of hyperparameter tuning. 
- Number of tumor and no-tumor images are very closely balanced by construction. The constraint on minimum relative cross-sectional area and the size of the grade 4 tumors did lead to a reduction in allowed no-tumor images at preprocessing.
- Tumor grade is heavily imbalanced, but not the target of the classifier. Splits are still stratified in tumor grade.
- A single DataFrame is generated using SampleMapper and contains all of the patients images from a given modality. This DataFrame is used to create subset DataFrames containing all of the images from patients that are included in the training/validation/test set. 
- The training/validation sets are stratified on tumor grade as well.
- The starting model is the resnet18 model in the torchvision library. 
- The fully-connected layer with 1000 categories is replaced with a fully connected block with a 1024 node hidden layer with ReLU activation to a single output with Sigmoid activation.
- To reduce overfitting, this new fully connected layer uses dropout regularization with a 50% dropout rate. 
- All of the residual block parameters are frozen during training. Only the fully connected layer is updated.
- Many modeling iterations led the model architecture described above. It could still be improved upon.
- The models are trained over 5 epochs due to the iteration time. 
- The 'best' model parameters from each split are determined based on the epoch with the highest validation set accuracy score with a 0.5 threshold.
- A more sophisticated approach should be used to determine early stopping of the training and a best set of model parameters.

Summary of Results:
- The 4-fold cross validation procedure yielded models with similar classification performance, with minor variations in the ROC curve and PR curve for one fold.
- ROC AUC of the models from each split were near 0.89 and average precision of the models around 0.9. 
- The distribution of scores for the positive class and the negative class were similar between folds. 
- Score distributions show overlap between classes. 
- The chosen models from each split are used to again check their validation accuracy score. 
- The model with the highest validation accuracy was chosen as the final model and used to evaluate the test dataset. 
- The model parameters are saved in with parameters saved in model_fold2_resnet_FLAIR.pth
- ROC AUC and average precision of the test set are consistent with the validation set. 
- Accuracy score of the model agianst the test set with a classification threshold of 0.5 is ~80%.
- Summary of results can be found in the Summarize_BC_ResNet_splits.ipynb notebook and in the figures directory.


models/model_foldX_resnet_FLAIR.pth:
- Best model from each fold (X).
- Final model taken from fold 2.
    
