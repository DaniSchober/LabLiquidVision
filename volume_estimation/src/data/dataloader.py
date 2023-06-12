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
                vessel_depth_path = os.path.join(
                    sample_path, "Input_EmptyVessel_Depth_segmented.npy"
                )
                liquid_depth_path = os.path.join(
                    sample_path, "Input_ContentDepth_segmented.npy"
                )
                vol_liquid_path = os.path.join(sample_path, "Input_vol_liquid.txt")
                vessel_path = os.path.join(sample_path, "Input_vessel.txt")
                vol_vessel_path = os.path.join(sample_path, "Input_vol_vessel.txt")
                image_path = os.path.join(sample_path, "Input_visualize.png")
                segmentation_vessel_path = os.path.join(
                    sample_path, "Input_VesselMask.npy"
                )
                segmentation_liquid_path = os.path.join(
                    sample_path, "Input_ContentMaskClean.npy"
                )
                depth_map_path = os.path.join(sample_path, "Input_DepthMap.npy")

                if (
                    os.path.exists(vessel_depth_path)
                    and os.path.exists(liquid_depth_path)
                    and os.path.exists(vol_liquid_path)
                    and os.path.exists(vessel_path)
                    and os.path.exists(vol_vessel_path)
                    and os.path.exists(image_path)
                    and os.path.exists(segmentation_vessel_path)
                    and os.path.exists(segmentation_liquid_path)
                    and os.path.exists(depth_map_path)
                ):
                    self.samples.append(
                        (
                            vessel_depth_path,
                            liquid_depth_path,
                            vol_liquid_path,
                            vessel_path,
                            vol_vessel_path,
                            image_path,
                            segmentation_liquid_path,
                            segmentation_vessel_path,
                            depth_map_path,
                        )
                    )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomRotation(degrees=(-10, 10)),  # Add random rotation
                # transforms.Normalize(mean=[0.5], std=[0.5])
                # Add other augmentation transforms here
            ]
        )

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
            segmentation_liquid_path,
            segmentation_vessel_path,
            depth_map_path,
        ) = self.samples[index]

        vessel_depth = np.load(vessel_depth_path).astype(np.float32)
        liquid_depth = np.load(liquid_depth_path).astype(np.float32)
        segmentation_liquid = np.load(segmentation_liquid_path).astype(np.float32)
        segmentation_vessel = np.load(segmentation_vessel_path).astype(np.float32)
        depth_map = np.load(depth_map_path).astype(np.float32)

        # convert vessel depth map from log to linear
        vessel_depth = np.exp(vessel_depth)

        # convert liquid depth map from log to linear
        liquid_depth = np.exp(liquid_depth)

        depth_map_vessel = depth_map * segmentation_vessel
        depth_map_liquid = depth_map * segmentation_liquid
        # get median of depth map of non-zero values
        ground_truth_depth_vessel = np.median(depth_map_vessel[depth_map_vessel != 0])
        ground_truth_depth_liquid = np.median(depth_map_liquid[depth_map_liquid != 0])

        # normalize vessel depth map of non zero values
        vessel_depth_normalized = vessel_depth.copy()
        # Create a mask to identify the non-zero values
        mask = vessel_depth != 0
        # Calculate the mean and standard deviation of the non-zero values
        mean = np.mean(vessel_depth[mask])
        std = np.std(vessel_depth[mask])
        # Normalize/standardize the non-zero values
        vessel_depth_normalized[mask] = (vessel_depth[mask] - mean + 10) / std

        # normalize liquid depth map of non zero values
        liquid_depth_normalized = liquid_depth.copy()
        # Create a mask to identify the non-zero values
        mask = liquid_depth != 0
        # Calculate the mean and standard deviation of the non-zero values
        mean = np.mean(liquid_depth[mask])
        std = np.std(liquid_depth[mask])
        # Normalize/standardize the non-zero values
        liquid_depth_normalized[mask] = (liquid_depth[mask] - mean + 10) / std

        # scale the depth maps to the ground truth depth
        vessel_depth_scaled = vessel_depth_normalized * ground_truth_depth_vessel
        liquid_depth_scaled = liquid_depth_normalized * ground_truth_depth_liquid


        # vessel_depth_masked = depth_map * segmentation_vessel
        # ground_truth_depth = vessel_depth_masked[vessel_depth_masked != 0].mean()

        # normalize the depth images
        # vessel_depth = vessel_depth - vessel_depth.mean()

        # liquid_depth = liquid_depth * ground_truth_depth
        # vessel_depth = vessel_depth * ground_truth_depth

        # vessel_depth = self.transform(vessel_depth)

        # convert from numpy matrix to tensor
        # vessel_depth = torch.from_numpy(vessel_depth)
        # vessel_depth = F.interpolate(vessel_depth.unsqueeze(0), size=(160, 214), mode='bilinear', align_corners=False)

        # vessel_depth = F.interpolate(vessel_depth.unsqueeze(0).unsqueeze(0), size=(160, 214), mode='bilinear', align_corners=False)

        # vessel_depth = vessel_depth.squeeze(0).squeeze(0)

        # camera_intrinsics =

        # decrease the size of the vessel depth image
        # vessel_depth = vessel_depth[::3, ::3]

        if self.transform == True:
            liquid_depth = self.transform(liquid_depth)
            liquid_depth = F.interpolate(
                liquid_depth.unsqueeze(0),
                size=(160, 214),
                mode="bilinear",
                align_corners=False,
            )
            vessel_depth = self.transform(vessel_depth)
            vessel_depth = F.interpolate(
                vessel_depth.unsqueeze(0),
                size=(160, 214),
                mode="bilinear",
                align_corners=False,
            )
        else:
            liquid_depth = torch.from_numpy(liquid_depth)
            liquid_depth = F.interpolate(
                liquid_depth.unsqueeze(0).unsqueeze(0),
                size=(160, 214),
                mode="bilinear",
                align_corners=False,
            )
            liquid_depth = liquid_depth.squeeze(0).squeeze(0)
            vessel_depth = torch.from_numpy(vessel_depth)
            vessel_depth = F.interpolate(
                vessel_depth.unsqueeze(0).unsqueeze(0),
                size=(160, 214),
                mode="bilinear",
                align_corners=False,
            )
            vessel_depth = vessel_depth.squeeze(0).squeeze(0)
            segmentation_liquid = torch.from_numpy(segmentation_liquid)
            segmentation_liquid = F.interpolate(
                segmentation_liquid.unsqueeze(0),
                size=(160, 214),
                mode="bilinear",
                align_corners=False,
            )
            segmentation_liquid = segmentation_liquid.squeeze(0).squeeze(0)
            segmentation_vessel = torch.from_numpy(segmentation_vessel)
            segmentation_vessel = F.interpolate(
                segmentation_vessel.unsqueeze(0),
                size=(160, 214),
                mode="bilinear",
                align_corners=False,
            )
            segmentation_vessel = segmentation_vessel.squeeze(0).squeeze(0)

            vessel_depth_scaled = torch.from_numpy(vessel_depth_scaled)
            vessel_depth_scaled = F.interpolate(
                vessel_depth_scaled.unsqueeze(0).unsqueeze(0),
                size=(160, 214),
                mode="bilinear",
                align_corners=False,
            )
            vessel_depth_scaled = vessel_depth_scaled.squeeze(0).squeeze(0)

            liquid_depth_scaled = torch.from_numpy(liquid_depth_scaled)
            liquid_depth_scaled = F.interpolate(
                liquid_depth_scaled.unsqueeze(0).unsqueeze(0),
                size=(160, 214),
                mode="bilinear",
                align_corners=False,
            )
            liquid_depth_scaled = liquid_depth_scaled.squeeze(0).squeeze(0)


        # convert from numpy matrix to tensor
        # liquid_depth = torch.from_numpy(liquid_depth)
        # liquid_depth = F.interpolate(liquid_depth.unsqueeze(0), size=(160, 214), mode='bilinear', align_corners=False)

        # liquid_depth = F.interpolate(liquid_depth.unsqueeze(0).unsqueeze(0), size=(160, 214), mode='bilinear', align_corners=False)

        # liquid_depth = liquid_depth.squeeze(0).squeeze(0)

        # decrease the size of the liquid depth image

        # liquid_depth = liquid_depth[::3, ::3]

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
            "segmentation_liquid": segmentation_liquid,
            "segmentation_vessel": segmentation_vessel,
            "vessel_depth_scaled": vessel_depth_scaled,
            "liquid_depth_scaled": liquid_depth_scaled,
        }
