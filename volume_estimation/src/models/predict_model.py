import torch
import argparse
from src.models.model import VesselNet
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--depth_image", type=str, help="Path to the depth map file")
args = parser.parse_args()

model = VesselNet()
model.load_state_dict(torch.load("models/vessel_net.pth"))


depth_image = np.load(args.depth_image).astype(np.float32)
# print(depth_image.shape)
depth_image = torch.from_numpy(depth_image)
# print(depth_image.shape)
depth_image = depth_image.unsqueeze(0)
# print(depth_image.shape)


with torch.no_grad():
    model.eval()
    outputs = model(depth_image)
    vol_liquid = outputs[0][0].item()
    vol_vessel = outputs[0][1].item()

print(f"Predicted liquid volume: {vol_liquid:.2f} ml")
print(f"Predicted vessel volume: {vol_vessel:.2f} ml")
