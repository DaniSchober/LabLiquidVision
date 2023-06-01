import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms



class VesselCaptureDataset(Dataset):
    def __init__(self, data_dir, transform=False):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        for sample_dir in os.listdir(data_dir):
            sample_path = os.path.join(data_dir, sample_dir)

            if os.path.isdir(sample_path):
                vessel_depth_path = os.path.join(sample_path, "Input_EmptyVessel_Depth_segmented.npy")
                liquid_depth_path = os.path.join(sample_path, "Input_ContentDepth_segmented.npy")
                vol_liquid_path = os.path.join(sample_path, "Input_vol_liquid.txt")
                vessel_path = os.path.join(sample_path, "Input_vessel.txt")
                vol_vessel_path = os.path.join(sample_path, "Input_vol_vessel.txt")
                image_path = os.path.join(sample_path, "Input_visualize.png")

                if (
                    os.path.exists(vessel_depth_path)
                    and os.path.exists(liquid_depth_path)
                    and os.path.exists(vol_liquid_path)
                    and os.path.exists(vessel_path)
                    and os.path.exists(vol_vessel_path)
                    and os.path.exists(image_path)
                ):
                    self.samples.append(
                        (
                            vessel_depth_path,
                            liquid_depth_path,
                            vol_liquid_path,
                            vessel_path,
                            vol_vessel_path,
                            image_path,
                        )
                    )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=(-10, 10)),  # Add random rotation
            #transforms.Normalize(mean=[0.5], std=[0.5])
            # Add other augmentation transforms here
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        (
            vessel_depth_path,
            liquid_depth_path,
            vol_liquid_path,
            vessel_path,
            vol_vessel_path,
            image_path,
        ) = self.samples[index]

        vessel_depth = np.load(vessel_depth_path).astype(np.float32)

        #vessel_depth = self.transform(vessel_depth)

        # convert from numpy matrix to tensor
        #vessel_depth = torch.from_numpy(vessel_depth)
        #vessel_depth = F.interpolate(vessel_depth.unsqueeze(0), size=(160, 214), mode='bilinear', align_corners=False)

        #vessel_depth = F.interpolate(vessel_depth.unsqueeze(0).unsqueeze(0), size=(160, 214), mode='bilinear', align_corners=False)

        #vessel_depth = vessel_depth.squeeze(0).squeeze(0)



        # decrease the size of the vessel depth image
        #vessel_depth = vessel_depth[::3, ::3]

        liquid_depth = np.load(liquid_depth_path).astype(np.float32)

        if self.transform == True:
            liquid_depth = self.transform(liquid_depth)
            liquid_depth = F.interpolate(liquid_depth.unsqueeze(0), size=(160, 214), mode='bilinear', align_corners=False)
            vessel_depth = self.transform(vessel_depth)
            vessel_depth = F.interpolate(vessel_depth.unsqueeze(0), size=(160, 214), mode='bilinear', align_corners=False)
        else:
            liquid_depth = torch.from_numpy(liquid_depth)
            liquid_depth = F.interpolate(liquid_depth.unsqueeze(0).unsqueeze(0), size=(160, 214), mode='bilinear', align_corners=False)
            liquid_depth = liquid_depth.squeeze(0).squeeze(0)
            vessel_depth = torch.from_numpy(vessel_depth)
            vessel_depth = F.interpolate(vessel_depth.unsqueeze(0).unsqueeze(0), size=(160, 214), mode='bilinear', align_corners=False)
            vessel_depth = vessel_depth.squeeze(0).squeeze(0)


        
        # convert from numpy matrix to tensor
        #liquid_depth = torch.from_numpy(liquid_depth)
        #liquid_depth = F.interpolate(liquid_depth.unsqueeze(0), size=(160, 214), mode='bilinear', align_corners=False)

        #liquid_depth = F.interpolate(liquid_depth.unsqueeze(0).unsqueeze(0), size=(160, 214), mode='bilinear', align_corners=False)

        #liquid_depth = liquid_depth.squeeze(0).squeeze(0)


        # decrease the size of the liquid depth image
        
        #liquid_depth = liquid_depth[::3, ::3]



        vol_liquid = int(open(vol_liquid_path, "r").read().strip())
        vessel_name = open(vessel_path, "r").read().strip()
        vol_vessel = int(open(vol_vessel_path, "r").read().strip())

        return {
            "vessel_depth": vessel_depth,
            "liquid_depth": liquid_depth,
            "vol_liquid": torch.tensor(vol_liquid),
            "vessel_name": vessel_name,
            "vol_vessel": torch.tensor(vol_vessel),
            "image_path": image_path,
        }
