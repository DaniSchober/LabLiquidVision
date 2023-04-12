import os
import numpy as np
import torch
from torch.utils.data import Dataset


class VesselCaptureDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = []

        for sample_dir in os.listdir(data_dir):
            sample_path = os.path.join(data_dir, sample_dir)

            if os.path.isdir(sample_path):
                color_image_path = os.path.join(sample_path, "ColorImage.npy")
                depth_image_path = os.path.join(sample_path, "DepthImage.npy")
                vol_liquid_path = os.path.join(sample_path, "vol_liquid.txt")
                vessel_path = os.path.join(sample_path, "vessel.txt")
                vol_vessel_path = os.path.join(sample_path, "vol_vessel.txt")

                if (
                    os.path.exists(color_image_path)
                    and os.path.exists(depth_image_path)
                    and os.path.exists(vol_liquid_path)
                    and os.path.exists(vessel_path)
                    and os.path.exists(vol_vessel_path)
                ):
                    self.samples.append(
                        (
                            color_image_path,
                            depth_image_path,
                            vol_liquid_path,
                            vessel_path,
                            vol_vessel_path,
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        (
            color_image_path,
            depth_image_path,
            vol_liquid_path,
            vessel_path,
            vol_vessel_path,
        ) = self.samples[index]

        color_image = np.load(color_image_path).astype(np.float32)
        depth_image = np.load(depth_image_path).astype(np.float32)
        vol_liquid = int(open(vol_liquid_path, "r").read().strip())
        vessel_name = open(vessel_path, "r").read().strip()
        vol_vessel = int(open(vol_vessel_path, "r").read().strip())

        return {
            "color_image": torch.from_numpy(color_image),
            "depth_image": torch.from_numpy(depth_image),
            "vol_liquid": torch.tensor(vol_liquid),
            "vessel_name": vessel_name,
            "vol_vessel": torch.tensor(vol_vessel),
        }
