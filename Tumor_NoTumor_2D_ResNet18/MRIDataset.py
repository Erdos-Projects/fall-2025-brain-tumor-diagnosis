import numpy as np
import nibabel as nib
import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from torchvision.transforms import v2
import PIL



class SampleMapper():
    def __init__(self,
                 data_dir: str,
                 modality: str) -> None:
        self.data_dir = data_dir
        self.modality = modality
        # Data directory expected to have 'tumor' and 'notumor' sub-directories
        self.data_map = pd.DataFrame([], columns=["patient_id", "path", "target"])
        # Traverse data_dir
        tumor_dir = self.data_dir + "/tumor/" + self.modality
        notumor_dir = self.data_dir + "/notumor/" + self.modality
        # name template is "{patient_id}_mri-axis-{mri_axis}_slice-{slice}.png"
        # patient_id matches 'UCSF-PDGM-XXXX' 
        for tumor_file in os.listdir(tumor_dir):
            file_split = tumor_file.split("_")
            patient_id = file_split[0]
            file_path = tumor_dir + "/" + tumor_file
            self.data_map.loc[len(self.data_map)] = [patient_id, file_path, 1]
        for notumor_file in os.listdir(notumor_dir):
            file_split = notumor_file.split("_")
            patient_id = file_split[0]
            file_path = notumor_dir + "/" + notumor_file
            self.data_map.loc[len(self.data_map)] = [patient_id, file_path, 0]
                


class MRIDataset(Dataset):
    def __init__(self, 
                 data_map: pd.DataFrame, 
                 pre_transforms: list[v2.Transform]=[v2.PILToTensor(),v2.Resize((128,128)),v2.ToDtype(torch.float32, scale=True)],
                 additional_transforms: list[v2.Transform]=[],
                 to_rgb: bool=True) -> None:
        """
        Dataset for 2D MRI slices

        Args:
            data_map: Pandas DataFrame with columns patient_id, path, target. 
                      patient_id is the patient ID. For this dataset they take the form UCSF-PDGM-XXXX.
                      path contains a single string directing to the location of an MRI image. 
                      target is either 0 (no-tumor) or 1 (tumor).
                      Can be constructed from SampleMapper. 
            pre_transforms: Transformations applied to the raw image data.
                            Default preprocessing converts image to a pytorch tensor, followed by a resizing to 128,128 required for resnet18, and a type conversion and rescaling to [0,1].
            additional_transforms: Any additional transformations applied to data before training (someting like random rotations, flips, or zooms).
            to_rgb: Boolean value. True is set to convert grayscale images to 3-channel 'rgb' by stacking the grayscale image three times. Required as input to resnet18 
        """
        # data_map to be populated with (patient_id, slice, target)
        # patient_id: ID designating the patient label. Is the prefix to the to the data directory (*_nifti) and the individual mri files.
        #             patient_ids in this project take on the form 'UCSF-PDGM-XXXX' where 'XXXX' is a unique patient number.
        # slice: The mri_axis coordinate used to take the cross-sectional slice of the MRI volume. 
        #        The number of slices taken from both non-tumor containing segments of the brain and tumor containing segments of the brain 
        #        is expected to be equal.
        # target: The truth label for whether there is a tumor (1) or no tumor (0) for that slice of the brain MRI.

        self.data_map = data_map
        self.pre_transforms = pre_transforms
        self.additional_transforms = additional_transforms
        self.transform = v2.Compose([*self.pre_transforms, *self.additional_transforms])
        self.to_rgb = to_rgb


        
    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        item_info = self.data_map.iloc[idx]
        path = item_info["path"]
        label = int(item_info["target"])
        pil_image = PIL.Image.open(path)
        if self.to_rgb:
            pil_image = torch.tensor(np.stack([pil_image, pil_image, pil_image], axis=0))
        img_tensor = self.transform(pil_image)

        return img_tensor, torch.tensor(np.array([label]), dtype=torch.float32)

    
if __name__ == "__main__":
    import os
    from torch.utils.data import DataLoader
    patient_id_list = []
    data_dir = "../processed-data/images"
    modality = "T1c"
    sample_mapper = SampleMapper(data_dir,
                                 modality)


    data_map = sample_mapper.data_map
    dataset = MRIDataset(data_map)
    print(len(dataset))
    print(dataset[0])
    print(dataset[1])
    dataloader = DataLoader(dataset, batch_size=64)
    print("Loaded! Running the dataloader.")
    for iteration in dataloader:
        X, Y = iteration
        print("Length of X input:", len(X))
        print("Length of Y output:", len(Y))