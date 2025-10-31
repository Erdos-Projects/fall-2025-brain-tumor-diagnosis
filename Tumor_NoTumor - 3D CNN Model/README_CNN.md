This file directory includes a 3D CNN model using the pytorch library. The structure is broken down as follows:

3D_CNN.ipynb:

- Includes all preprocessing of our MRI data obtained via the Cancer Imaging Archive (UCSF PDGM)
- Preprocessing includes eliminating redundant patients, ignoring patients missing certain images and normalizing intensities of each MRI scan. 
- Also includes inclusion of metadata into NN, so splits of train/test/val can include accurate representation of tumor grades in attempt to limit bias from our model

- Using pytorch library, we segment the entire MRI image into smaller 3D subregions, either including a tumor and labeling (1) or selecting a tumor free region (0). 
- This has the added benefit of increasing our dataset, since a single patient can have multiple healthy and unhealthy scans for training.
- The model is then trained based on our 70/15/15 with our training and validation data taken to be a (96^3) voxel volume and our test data taken to be the entire MRI brain voxel volume (240^3). 
- The model is trained on a small subregion of the brain. Our goal is to have the model test the entire brain. To overcome this issue, we have created a 3D sliding window function to have the model 
  process sequences of subregions of the tested brain, and then stitch all subregions together to recreate the final brain image. In addition to reducing computational load, this has the benefit of
  allowing us to train on subsets of the 3D images, while testing on the entire brain.
- The model was trained over 20 epochs. The epoch with the best loss against our validation data, chosen to be a combination of Binary cross entropy (BCE) and Dice loss, is taken to be our model. 
- Output of the model is a tensor of shape [1,1,X,Y,Z], and is a 3D probability map of the likelyhood that each individual voxel in the brain image contains a tumor. For a given coordinate [X,Y,Z]
  in the brain, the model assigns an output between 0 and 1 of the likelyhood of that voxel being tumorous. Our criteria or threshold for deciding what probability constitutes returning (yes) for a tumor 
  is determined by running various thresholds agains the 3D probability map and computing what threshold returns the best overall loss for the test dataset. This threshold was ultimately set to 0.735.

Summary of Results:
- When evaluating against our test dataset (N = 25), the majority of our images scored above 0.8 against our Dice metric ( Dice = 1 indicates perfect overlap of established ground truth mask and predicted tumor region). 
  The results that scored poorly were visually inspected and seen to typically be images of poor quality or nonstandard contrast/brightness. Images of our probability map overlaid onto the MRI images can be seen towards the 
  end of the file for both the best and worst case scenarios. 
- Images of our results against all metrics can be seen further in the folder titled 'Summary Images'
- Overall the model responds well to various tumor grades but since the dataset only includes tumorous patients, we cannot say how well the model works on a full scan of a health brain. However, when looking at healthy subregions
  of tumorous patients, the model performs very well against all metrics.
  
best.pt:
- Model obtained using the code described above. Model formatted as a pytorch object and is used for all evaluation against test dataset.

test_metrics.csv:
- Output file summarizing scores of patient against each metric chosen.
- Shows the number of voxels that are classified as either true positive (TP), false positive (FP), true negative (NP), and false negative (FN).

test_metrics_thr_sweep.csv:
- Output file indicating what probability threshold will determine whether to classify a voxel as healthy or tumorous. 
- The best threshold value is chosen to be the one which maximizes the Dice metric.
    
UCSF-PDGM-metadata_v5.csv:
- Includes metadata of our chosen dataset.
- Includes things like tumor type and grade, as well as information about specific patients like Age or Sex (M/F).


