import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import glob
import logging
import torch

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec

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
from monai.utils import set_determinism

"""Validates MONAI networks for DL CBCT.

method used:
Denoising UNet

with functions modified as if:
sino = False 
low = False
"""

logger = logging.getLogger("Denoising")
logging.basicConfig(level=logging.INFO)

class Loader(Operator):
    def __init__(self, *args, **kwargs):
        self.data_dir = DATAPATH
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out_loader")

    def get_val_data(self, data_dir):
        """Gets val data
        """
        val_fdk_clean = sorted(glob.glob(os.path.join(data_dir, "validate", "*_clean_fdk_256.npy")))
        val_fdk_low = sorted(glob.glob(os.path.join(data_dir, "validate", "*_fdk_low_dose_256.npy")))
        val_fdk_clinical = sorted(glob.glob(os.path.join(data_dir, "validate", "*_fdk_clinical_dose_256.npy")))
        val_sino_low = sorted(glob.glob(os.path.join(data_dir, "validate", "*_sino_low_dose.npy")))
        val_sino_clinical = sorted(glob.glob(os.path.join(data_dir, "validate", "*_sino_clinical_dose.npy")))
        val_files = [{"fdk_clean": fdk_clean, "fdk_low": fdk_low, "fdk_clinical": fdk_clinical, "sino_low": sino_low, "sino_clinical": sino_clinical} for
                     fdk_clean, fdk_low, fdk_clinical, sino_low, sino_clinical, in zip(val_fdk_clean, val_fdk_low, val_fdk_clinical, val_sino_low, val_sino_clinical)]

        print(f"len(val_files)={len(val_files)}")
        
        return val_files

    def compute(self, op_input, op_output, context):
        print("Load")

        # Get transform & Cache Data
        val_files = self.get_val_data(self.data_dir)
        
        val_transforms = Compose([LoadImaged(keys=["fdk_clean", "fdk_low", "fdk_clinical"]),
                                  EnsureChannelFirstd(keys=["fdk_clean", "fdk_low", "fdk_clinical"])])
        
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

        # val_files, val_transforms, val_ds, val_loader
        to_send = [val_files, val_transforms, val_ds, val_loader]
        op_output.emit(to_send, "out_loader")

        
class Denoiser(Operator):
    def __init__(self, *args, **kwargs):
        self.device = torch.device("cuda:0")
        self.save_dir = os.path.join(OUTPUTPATH,"img/")
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in_loader")
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
        total_params = sum(p.numel() for p in model_img.parameters() if p.requires_grad)
        print(f"total_params={total_params:,}")

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

    def get_batch(self, batch_data, device):
        """Gets a data batch
        """
        inputs, targets = (
            batch_data["fdk_clinical"].to(device),
            batch_data["fdk_clean"].to(device),
        )

        return inputs, targets

    def forward(self, inputs, model_img):
        """Method Used:
        * Denoising UNet
        """
        return model_img(inputs)
    
    def compute(self, op_input, op_output, context):
        print("Denoising")
        # val_files, val_transforms, val_ds, val_loader
        val_files, val_transforms, val_ds, val_loader = op_input.receive("in_loader")

        # Get model, metric
        model_img = self.get_model(self.device)
        model_img.eval()

        metric = MSEMetric()

        # Get best learned model
        model_img = self.load_model(model_img, self.save_dir)
        model_img.eval()

        # Compute mean validation metric
        metrics = []
        with torch.no_grad():
            for batch_data in val_loader:
                print("*", end="")
                inputs, targets = self.get_batch(batch_data, self.device)
                
                print("forward")
                with torch.cuda.amp.autocast():
                    pred = self.forward(inputs, model_img)
                print("forward passed")
                
                mb = metric(y_pred=pred, y=targets)
                metrics.append(float(mb))

            m = metric.aggregate().item()
            metric.reset()

        print(f"metric (n={len(val_loader.dataset)}) = {m:.6f}")

        metrics = np.asarray(metrics)
        #np.save("data.npy", metrics)
        
        # val_loader, model_img, metric
        to_send = [val_loader, model_img, metric]
        op_output.emit(to_send, "out_denoiser")
       
        
class Plotter(Operator):
    def __init__(self, *args, **kwargs):
        self.device = torch.device("cuda:0")
        self.save_dir = os.path.join(OUTPUTPATH,"img/")
        super().__init__(*args, **kwargs)
    
    def setup(self, spec: OperatorSpec):
        spec.input("in_denoiser")

    def get_batch(self, batch_data, device):
        """Gets a data batch
        """
        inputs, targets = (
            batch_data["fdk_clinical"].to(device),
            batch_data["fdk_clean"].to(device),
        )

        return inputs, targets

    def forward(self, inputs, model_img):
        """Method used:
        * Denoising UNet
        """
        return model_img(inputs)

    def show_validation_fit(self, inputs, targets, pred, metric, save_dir, show=False, save_fig=True, slice_ix=128):
        """Plots validation fit of trained model (as image)
        """
        plt.figure("check", (10, 6))
        plt.subplot(1, 3, 1)
        plt.title(f"targets")
        plt.imshow(targets[0, 0, slice_ix, :, :].cpu(), cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title(f"inputs")
        plt.imshow(inputs[0, 0, slice_ix, :, :].cpu(), cmap="gray")
        plt.subplot(1, 3, 3)
        plt.title(f"pred, metric={metric:0.6f}")
        plt.imshow(pred.detach().cpu()[0, 0, slice_ix, :, :], cmap="gray")
        plt.axis('off')
        if save_fig:
            plt.savefig(os.path.join(save_dir, "validation_fit_clinical.png"))
        if show:
            plt.show()
        
    def compute(self, op_input, op_output, context):
        print("Plot")
        
        #val_loader, model_img, metric
        val_loader, model_img, metric = op_input.receive("in_denoiser")

        check_ix = 0
        with torch.no_grad():
            for i, batch_data in enumerate(val_loader):
                if i != check_ix: continue
                inputs, targets = self.get_batch(batch_data, self.device)
            
                print("forward")
                pred = self.forward(inputs, model_img)
                print("forward passed")
            
                metric(y_pred=pred, y=targets)
                m = metric.aggregate().item()
                metric.reset()
                self.show_validation_fit(inputs, targets, pred, m, self.save_dir, show=True, save_fig=False)

                break

            
DATAPATH = "/cbct-recon/holohub/applications/denoising/python/data/"
OUTPUTPATH = "/cbct-recon/holohub/applications/denoising/python/output/"
class Denoising(Application):
    def compose(self):
        loader = Loader(self, CountCondition(self, 1), name="loader")
        denoiser = Denoiser(self, name="denoiser")
        plotter = Plotter(self, name="plotter")

        # Connect the operators into the workflow: load -> denoising
        self.add_flow(loader, denoiser)
        # Connect the operators into the workflow: denoising -> plot
        self.add_flow(denoiser, plotter)


if __name__ == "__main__":
    app = Denoising()
    app.run()
