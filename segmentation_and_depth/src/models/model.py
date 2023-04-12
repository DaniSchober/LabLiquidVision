#  Fully convolutional net that receive image and predict depth maps and segmentation maps.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Net(nn.Module):
    def __init__(
        self, MaskList, DepthList
    ):  # MaskList is list of segmentation mask to predict, DepthList is list of Depth map to predict

        # Build layers for standard FCN with only image as input
        super(Net, self).__init__()

        # Load pretrained  encoder
        self.Encoder = models.resnet101(weights="ResNet101_Weights.DEFAULT")
        
        # Dilated convolution ASPP layers (same as deep lab)
        self.ASPPScales = [1, 2, 4, 12, 16]
        self.ASPPLayers = nn.ModuleList()
        for scale in self.ASPPScales:
            self.ASPPLayers.append(
                nn.Sequential(
                    nn.Conv2d(
                        2048,
                        512,
                        stride=1,
                        kernel_size=3,
                        padding=(scale, scale),
                        dilation=(scale, scale),
                        bias=False,
                    ),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                )
            )

        # Squeeze ASPP Layer
        self.SqueezeLayers = nn.Sequential(
            nn.Conv2d(2560, 512, stride=1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # Skip conncetion layers for upsampling
        self.SkipConnections = nn.ModuleList()
        self.SkipConnections.append(
            nn.Sequential(
                nn.Conv2d(1024, 512, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
        )
        self.SkipConnections.append(
            nn.Sequential(
                nn.Conv2d(512, 256, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
        )
        self.SkipConnections.append(
            nn.Sequential(
                nn.Conv2d(256, 128, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
        )

        # Skip connection squeeze applied to the (concat of upsample+skip connecection layers)
        self.SqueezeUpsample = nn.ModuleList()
        self.SqueezeUpsample.append(
            nn.Sequential(
                nn.Conv2d(1024, 512, stride=1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
        )
        self.SqueezeUpsample.append(
            nn.Sequential(
                nn.Conv2d(
                    256 + 512, 256, stride=1, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
        )
        self.SqueezeUpsample.append(
            nn.Sequential(
                nn.Conv2d(
                    256 + 128, 256, stride=1, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
        )

        # Final prediction Depth maps
        #######################
        #######################
        # Dani: change this to depth prediction like in github for Sun3D
        #######################
        #######################
        self.OutLayersList = nn.ModuleList()
        self.OutLayersDicDepth = {}
        self.OutLayersDicMask = {}
        for nm in DepthList:
            self.OutLayersDicDepth[nm] = nn.Conv2d(
                256, 1, stride=1, kernel_size=3, padding=1, bias=False
            )
            self.OutLayersList.append(self.OutLayersDicDepth[nm])

        # Final prediction segmentation Mask
        self.OutLayersDicMask = {}
        for nm in MaskList:
            self.OutLayersDicMask[nm] = nn.Conv2d(
                256, 2, stride=1, kernel_size=3, padding=1, bias=False
            )
            self.OutLayersList.append(self.OutLayersDicMask[nm])

    def forward(
        self,
        Images,
        UseGPU=False,
        TrainMode=True,
        PredictDepth=True,
        PredictMasks=True,
        FreezeBatchNorm_EvalON=False,
    ):

        # Convert image to pytorch and normalize values
        RGBMean = [123.68, 116.779, 103.939]
        RGBStd = [65, 65, 65]

        if TrainMode == True:
            tp = torch.FloatTensor  # Training mode
        else:
            tp = torch.half  # Evaluation mode
            self.half()

        if FreezeBatchNorm_EvalON:
            self.eval()  # dont update batch 

        # Convert input to pytorch
        InpImages = (
            torch.autograd.Variable(
                torch.from_numpy(Images.astype(np.float32)), requires_grad=False
            )
            .transpose(2, 3)
            .transpose(1, 2)
            .type(tp)
        )

        # Convert to cuda if needed
        if UseGPU:
            InpImages = InpImages.cuda()
            self.cuda()
        else:
            InpImages = InpImages.cpu().float()
            self.cpu().float()

        # Normalize image values
        for i in range(len(RGBMean)):
            InpImages[:, i, :, :] = (InpImages[:, i, :, :] - RGBMean[i]) / RGBStd[
                i
            ]
        x = InpImages

        SkipConFeatures = []  # Store features map of layers used for skip connection
        # ---------------Run Encoder-----------------------------------------------------------------------------------------------------
        x = self.Encoder.conv1(x)
        x = self.Encoder.bn1(x)
        x = self.Encoder.relu(x)
        x = self.Encoder.maxpool(x)
        x = self.Encoder.layer1(x)
        SkipConFeatures.append(x)
        x = self.Encoder.layer2(x)
        SkipConFeatures.append(x)
        x = self.Encoder.layer3(x)
        SkipConFeatures.append(x)
        EncoderMap = self.Encoder.layer4(x)

        # ASPP Layers (Dilated conv)
        ASPPFeatures = []  # Results of various of scaled procceessing
        for ASPPLayer in self.ASPPLayers:
            y = ASPPLayer(EncoderMap)
            ASPPFeatures.append(y)
        x = torch.cat(ASPPFeatures, dim=1)
        x = self.SqueezeLayers(x)

        # Upsample features map  and combine with layers from encoder using skip connections
        for i in range(len(self.SkipConnections)):
            sp = (SkipConFeatures[-1 - i].shape[2], SkipConFeatures[-1 - i].shape[3])
            x = nn.functional.interpolate(
                x, size=sp, mode="bilinear", align_corners=False
            )  # Upsample
            x = torch.cat(
                (self.SkipConnections[i](SkipConFeatures[-1 - i]), x), dim=1
            )  # Apply skip connection and concat with upsample
            x = self.SqueezeUpsample[i](x)  # Squeeze

        # Final Depth map prediction
        #####################
        # Dani: Changed this to prediction of depth maps (see: Github Sun3D)
        #####################
        #####################
        self.OutDepth = {}
        if PredictDepth:
            for nm in self.OutLayersDicDepth:
                # print(nm)
                l = self.OutLayersDicDepth[nm](x)
                if (
                    TrainMode == False
                ):  # For prediction mode resize to the input image size
                    l = nn.functional.interpolate(
                        l,
                        size=InpImages.shape[2:4],
                        mode="bilinear",
                        align_corners=False,
                    )  # Resize to original image size
                self.OutDepth[nm] = l
        
        # Output segmentation mask
        self.OutProbMask = {}
        self.OutMask = {}
        if PredictMasks:
            for nm in self.OutLayersDicMask:
                l = self.OutLayersDicMask[nm](x)
                if (
                    TrainMode == False
                ):  # For prediction mode resize to the input image size
                    l = nn.functional.interpolate(
                        l,
                        size=InpImages.shape[2:4],
                        mode="bilinear",
                        align_corners=False,
                    )  # Resize to original image size
                Prob = F.softmax(l, dim=1)  # Calculate class probability per pixel
                tt, Labels = l.max(1)  # Find label per pixel
                self.OutProbMask[nm] = Prob
                self.OutMask[nm] = Labels
        return self.OutDepth, self.OutProbMask, self.OutMask
