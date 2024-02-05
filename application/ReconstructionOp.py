import os
from argparse import ArgumentParser

import numpy as np
import tomosipo as ts
import torch
from monai.data import CacheDataset, DataLoader, Dataset, ArrayDataset
from monai.transforms import Compose, ToTensor
from holoscan.core import Application, Operator, OperatorSpec

from ts_algorithms import fdk

from datetime import datetime

from PARAM import *
class ReconstructionOp(Operator):
    def __init__(self, *args, **kwargs):
        self.data_dir = os.path.join(PATH,"data", "validate/")
        super().__init__(*args, **kwargs)
        
    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out_reconstructor")

    def reconstruct(self, sino):
        image_size = [300, 300, 300]
        image_shape = [256, 256, 256]
        voxel_size = [1.171875, 1.171875, 1.171875]
        detector_shape = [256, 256]
        detector_size = [600, 600]
        pixel_size = [2.34375, 2.34375]
        dso = 575
        dsd = 1050
        angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
        sino=torch.from_numpy(sino).cuda()

        # Create a tomosipo operator
        vg = ts.volume(shape=image_shape, size=image_size)
        pg = ts.cone(angles=angles, shape=detector_shape, size=detector_size, src_orig_dist=dso, src_det_dist=dsd)
        A = ts.operator(vg, pg)

        recon= fdk(A, sino).cpu().numpy()
        
        return recon

    def save_reconstructed(self, file_path, recon):
        file_path = os.path.join(file_path, "reconstructed.npy")
        np.save(file_path, recon)
        #print(f"Reconstruction successfully saved to {file_path}")
        
    def compute(self, op_input, op_output, context):
        start_time = datetime.now()
        print("Reconstruction")
        sino_np = op_input.receive("in")
        recon = self.reconstruct(sino_np)
        to_save_path = self.data_dir
        if (COMPARISON):
            self.save_reconstructed(to_save_path, recon)
        #print(recon.shape)
        to_send = recon
        op_output.emit(to_send, "out_reconstructor")
        print("Reconstruction Time:", datetime.now() - start_time)
