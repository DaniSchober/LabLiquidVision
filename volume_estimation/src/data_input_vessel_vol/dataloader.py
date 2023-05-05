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
                vessel_depth_path = os.path.join(sample_path, "Input_EmptyVessel_Depth_segmented.npy")
                liquid_depth_path = os.path.join(sample_path, "Input_ContentDepth_segmented.npy")
                vol_liquid_path = os.path.join(sample_path, "Input_vol_liquid.txt")
                vessel_path = os.path.join(sample_path, "Input_vessel.txt")
                vol_vessel_path = os.path.join(sample_path, "Input_vol_vessel.txt")

                if (
                    os.path.exists(vessel_depth_path)
                    and os.path.exists(liquid_depth_path)
                    and os.path.exists(vol_liquid_path)
                    and os.path.exists(vessel_path)
                    and os.path.exists(vol_vessel_path)
                ):
                    self.samples.append(
                        (
                            vessel_depth_path,
                            liquid_depth_path,
                            vol_liquid_path,
                            vessel_path,
                            vol_vessel_path,
                        )
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
        ) = self.samples[index]

        vessel_depth = np.load(vessel_depth_path).astype(np.float32)
        # decrease the size of the vessel depth image
        #vessel_depth = vessel_depth[::2, ::2]

        liquid_depth = np.load(liquid_depth_path).astype(np.float32)
        vol_liquid = int(open(vol_liquid_path, "r").read().strip())
        vessel_name = open(vessel_path, "r").read().strip()
        vol_vessel = int(open(vol_vessel_path, "r").read().strip())

        return {
            "vessel_depth": torch.from_numpy(vessel_depth),
            "liquid_depth": torch.from_numpy(liquid_depth),
            "vol_liquid": torch.tensor(vol_liquid),
            "vessel_name": vessel_name,
            "vol_vessel": torch.tensor(vol_vessel),
        }
