import torch

'''
This file contains the loss functions used to train the model for depth estimation
'''

def DepthLoss(output, target, ROI):
    '''
    output: [B, 1, H, W]
    target: [B, 1, H, W]
    ROI: [B, 1, H, W]

    return: depth loss

    '''
    # https://proceedings.neurips.cc/paper/2014/file/7bccfde7714a1ebadf06c5f4cea752c1-Paper.pdf
    n = torch.sum(ROI, (1, 2, 3)) + 0.01  # Number of pixel in image
    di = (target - output) * ROI
    di2 = torch.pow(di, 2)
    DifTerm = torch.sum(di2, (1, 2, 3)) / n
    ScaleTerm = torch.pow(torch.sum(di, (1, 2, 3)), 2) / (n**2)
    loss = DifTerm - ScaleTerm
    return loss.mean()  # To block gradient turn to numpy and back to pytorch
