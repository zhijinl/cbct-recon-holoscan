import os

import matplotlib.pyplot as plt
import numpy as np
import glob
import logging
import tomosipo as ts
import torch

from holoscan.core import Application, Operator, OperatorSpec
from argparse import ArgumentParser

from PARAM import *
class LoadFileOp(Operator):
    def __init__(self, *args, **kwargs):
        self.data =  os.path.join(PATH,"data","0000_sino_clinical_dose.npy")
        super().__init__(*args, **kwargs)
        
    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def load_sinogram(self, file_path):
        # Load sinogram
        sino=np.load(file_path, allow_pickle=True)
        #print("Sinogram loaded from " + file_path)
        return sino

    def compute(self, op_input, op_output, context):
        print("Load Sinogram")
        path = self.data
        sino = self.load_sinogram(path)
        # output the reconstructed numpy array
        op_output.emit(sino, "out")
