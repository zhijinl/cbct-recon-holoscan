import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec

from PARAM import *
class PlotOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def setup(self, spec: OperatorSpec):
        spec.input("in_denoiser")
        
    def compute(self, op_input, op_output, context):
        print("Plot")
        
        npy_array = op_input.receive("in_denoiser")
        plt.figure("check", (10, 6))
        plt.title(f"pred")
        plt.imshow(npy_array[128, :, :], cmap="gray")
        plt.show()
