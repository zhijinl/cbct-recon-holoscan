import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import glob
import logging
import tomosipo as ts
import torch

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec

from monai.transforms import Compose, ToTensor
from monai.data import CacheDataset, DataLoader, Dataset
from monai.networks.layers import Norm
from monai.metrics import MSEMetric
from monai.networks.nets import DynUNet, UNet, SegResNet
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    RandAffined,
)

from datetime import datetime

from PARAM import *
class DenoiserOp(Operator):
    def __init__(self, *args, **kwargs):
        self.device = torch.device("cuda:0")
        self.save_dir = os.path.join(PATH,"output","img/")
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in_reconstructor")
        spec.output("out_denoiser")
        
    def get_model(self, device):
        """Gets model(s) for each method
        """
        model_img = SegResNet(
            spatial_dims=3,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=32,
            in_channels=1,
            out_channels=1,
            dropout_prob=0.2,
        ).to(device)

        return model_img
    
    def load_model(self, model_img, save_dir, tag="best_metric"):
        """Loads model(s)
        """
        model_img.load_state_dict(
            torch.load(
                os.path.join(save_dir, f"{tag}_img_clinical.pth")
            )
        )
            
        return model_img

    def forward(self, inputs, model_img):
        """Method Used:
        * Denoising UNet
        """
        return model_img(inputs)

    
    def compute(self, op_input, op_output, context):
        start_time = datetime.now()
        
        print("Denoising")
        
        # recon
        recon = op_input.receive("in_reconstructor")
        reconTensor = recon[None,None]
        transform = Compose([ToTensor()])
        recon = transform(reconTensor).to(self.device)

        # Get model, metric
        model_img = self.get_model(self.device)
        model_img.eval()

        metric = MSEMetric()

        # Get best learned model
        model_img = self.load_model(model_img, self.save_dir)
        model_img.eval()

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = self.forward(recon, model_img)

        npy_array = pred.detach().cpu().numpy().squeeze(0).squeeze(0)
        
        #npy_array
        to_send = npy_array
        op_output.emit(to_send, "out_denoiser")
        
        print("Denoising Time:", datetime.now() - start_time)
       
