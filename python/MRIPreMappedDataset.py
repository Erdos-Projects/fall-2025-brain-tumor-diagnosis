import numpy as np
import nibabel as nib
import pandas as pd
import torch
import os
from torch.utils.data import Dataset



class SampleMapper():
    def __init__(self,
                 data_dir: str,
                 patient_id_list: list[str],
                 samples_per_patient_per_label: int = 2,
                 min_relative_brain_area_per_sample: float = .25,
                 mri_axis=2,
                 random_seed=360):
        self.data_dir = data_dir
        self.patient_id_list = patient_id_list
        self.samples_per_patient_per_label = samples_per_patient_per_label
        self.min_relative_brain_area_per_sample = min_relative_brain_area_per_sample
        self.mri_axis = mri_axis
        self.transverse_axes = tuple([axis for axis in [0,1,2] if axis != self.mri_axis])
        self.random_seed = random_seed
        self.random_sampler = np.random.default_rng(seed=self.random_seed)

        self.data_map = pd.DataFrame([], columns=["patient_id", "slice", "target"])

        for patient_id in self.patient_id_list:
            patient_directory = f"{self.data_dir}/{patient_id}_nifti"
            brain_seg_path = f"{patient_directory}/{patient_id}_brain_segmentation.nii.gz"
            tumor_seg_path = f"{patient_directory}/{patient_id}_tumor_segmentation.nii.gz"
            brain_seg_mri = nib.load(brain_seg_path).get_fdata() # 3D array containing brian segmentation
            tumor_seg_mri = nib.load(tumor_seg_path).get_fdata() # 3D array containing tumor segmentation
            normalized_brain_seg_area = brain_seg_mri.sum(axis=self.transverse_axes) / np.max(brain_seg_mri.sum(axis=self.transverse_axes))
            above_min_relative_brain_area_mask = normalized_brain_seg_area > self.min_relative_brain_area_per_sample
            tumor_location_mask = tumor_seg_mri.any(axis=self.transverse_axes)
            min_relative_area_no_tumor_mask = np.logical_and(above_min_relative_brain_area_mask, ~tumor_location_mask)
            if tumor_location_mask.sum() < self.samples_per_patient_per_label or min_relative_area_no_tumor_mask.sum() < self.samples_per_patient_per_label:
                n_samples = min(tumor_location_mask.sum(), min_relative_area_no_tumor_mask.sum())
                print(f"{patient_id} has too few acceptable slices along the chosen axis.")
                print(f"\tSlices with tumors: {tumor_location_mask.sum()}")
                print(f"\tSlices with greater than minimum brian area: {above_min_relative_brain_area_mask.sum()}")
                print(f"\tSlices with greater than minimum brain area without any tumor material: {min_relative_area_no_tumor_mask.sum()}")
                print(f"\tSamples taken from {patient_id} have been reduced to {n_samples} from {self.samples_per_patient_per_label}.")
                if n_samples == 0:
                    print(f"Patient {patient_id} will not be included in the dataset.")
            else:
                n_samples = self.samples_per_patient_per_label
            indices_along_axis = np.arange(tumor_location_mask.size)
            min_relative_area_no_tumor_indices = indices_along_axis[min_relative_area_no_tumor_mask]
            tumor_location_indices = indices_along_axis[tumor_location_mask]
            random_tumorless_brain_slices = self.random_sampler.choice(min_relative_area_no_tumor_indices, size=n_samples, replace=False)
            random_tumor_slices = self.random_sampler.choice(tumor_location_indices, size=n_samples, replace=False)
            for slice in random_tumorless_brain_slices:            
                self.data_map.loc[len(self.data_map)] = [patient_id, slice, 0]
            for slice in random_tumor_slices:
                self.data_map.loc[len(self.data_map)] = [patient_id, slice, 1]
                


class MRIPreMappedDataset(Dataset):
    def __init__(self, 
                 base_path: str, 
                 data_map: pd.DataFrame, 
                 selected_modalities: list[str] = ["T1"],
                 mri_axis: int = 2) -> None:
        """
        Dataset for multi-channel 2D MRI slices

        Args:
            base_path: Path to folder containing patient folders.
            data_map: Dataframe with rows of patient_id (str), slice (int), and target (0 or 1).
            selected_modalities: List of modality names to use. Default to T1.
            mri_axis: The longitudinal axis images are selected from. Must be an integer value from 0, 1, 2.
        """
        # data_map to be populated with (patient_id, slice, target)
        # patient_id: ID designating the patient label. Is the prefix to the to the data directory (*_nifti) and the individual mri files.
        #             patient_ids in this project take on the form 'UCSF-PDGM-XXXX' where 'XXXX' is a unique patient number.
        # slice: The mri_axis coordinate used to take the cross-sectional slice of the MRI volume. 
        #        The number of slices taken from both non-tumor containing segments of the brain and tumor containing segments of the brain 
        #        is expected to be equal.
        # target: The truth label for whether there is a tumor (1) or no tumor (0) for that slice of the brain MRI.

        self.base_path = base_path
        self.data_map = data_map
        self.selected_modalities = selected_modalities
        self.mri_axis = mri_axis
        self.transverse_axes = tuple([axis for axis in [0,1,2] if axis != self.mri_axis])
        if self.mri_axis not in (0,1,2):
            print("Warning: mri_axis is out of bounds. Please select an integer from 0, 1, and 2. Setting mri_axis to 2 for this instance.")
            self.mri_axis = 2
            self.transverse_axes = (0,1)





        
    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        item_info = self.data_map.iloc[idx]
        patient_id = item_info["patient_id"]
        slice = item_info["slice"]
        target = item_info["target"]
        patient_directory = f"{self.base_path}/{patient_id}_nifti"        
        channels = []
        for modality in self.selected_modalities:
            modality_path = f"{patient_directory}/{patient_id}_{modality}.nii.gz"
            try:
                # Load 3D data
                img_3d = nib.load(modality_path).get_fdata().astype(np.float32)
                
                # Extract 2D slice
                if self.mri_axis == 0:
                    img_2d = img_3d[slice, :, :].copy()
                elif self.mri_axis == 1:
                    img_2d = img_3d[:, slice, :].copy()
                elif self.mri_axis == 2:
                    img_2d = img_3d[:, :, slice].copy()
                else:
                    print("Warning: mri_axis out of bounds (0,1,2). Using mri_axis=2")
                    img_2d = img_3d[:, :, slice].copy()
                # Normalize to [0, 1]
                img_min, img_max = img_2d.min(), img_2d.max()
                if img_max > img_min:
                    img_2d = (img_2d - img_min) / (img_max - img_min)
                else:
                    img_2d = np.zeros_like(img_2d)

                channels.append(img_2d)

            except Exception as e:
                # If file missing or error, use zero-filled channel
                print(f"Warning: Could not load {modality} for {patient_directory}: {e}")
                print(f"Filling channel with zeros.")
                channels.append(np.zeros((240, 240), dtype=np.float32))

        # Stack channels: (num_channels, H, W)
        img_tensor = np.stack(channels, axis=0)

        # Get label
        label = torch.tensor(target, dtype=torch.float32)

        return torch.tensor(img_tensor, dtype=torch.float32), torch.tensor(np.array([label]), dtype=torch.float32)

    
if __name__ == "__main__":
    import os
    from torch.utils.data import DataLoader
    patient_id_list = []
    data_dir = "../data"
    for patient in os.listdir(data_dir):
        if "FU" in patient:
            continue
        patient_id_list.append(patient.replace("_nifti", ""))

    sample_mapper = SampleMapper(data_dir,
                                 patient_id_list=patient_id_list,
                                 samples_per_patient_per_label=2,
                                 min_relative_brain_area_per_sample=.25,
                                 mri_axis=2,
                                 random_seed=360)


    data_map = sample_mapper.data_map
    mri_axis = sample_mapper.mri_axis
    selected_modalities = ["T1"]
    dataset = MRIPreMappedDataset(data_dir, data_map, selected_modalities=selected_modalities, mri_axis=mri_axis)
    print(len(dataset))
    print(dataset[0])
    print(dataset[1])
    dataloader = DataLoader(dataset, batch_size=64)
    print("Loaded! Running the dataloader.")
    for iteration in dataloader:
        X, Y = iteration
        print("Length of X input:", len(X))
        print("Length of Y output:", len(Y))