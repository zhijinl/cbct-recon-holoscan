import os
import logging

from holoscan.conditions import CountCondition
from holoscan.core import Application

from LoadFileOp import LoadFileOp
from ReconstructionOp import ReconstructionOp
from DenoiserOp import DenoiserOp
from PlotOp import PlotOp
from ConvertOp import ConvertOp

"""Validates MONAI networks for DL CBCT.

method used:
Denoising UNet

with functions modified as if:
sino = False 
low = False
"""

logger = logging.getLogger("Denoising")
logging.basicConfig(level=logging.INFO)

class cbct_reconstruction(Application):
    def compose(self):
        # the input path is already in load_op
        load_op = LoadFileOp(self, CountCondition(self, 1), name="load_op")
        rec_op = ReconstructionOp(self, name="rec_op")
        
        converterReconstructed = ConvertOp(self, name="converterReconstructed", patientName="Reconstructed")
        
        denoiser = DenoiserOp(self, name="denoiser")
        converterDenoised = ConvertOp(self, name="converterDenoised", patientName="Denoised")

        
        # Define the workflow: load_op -> rec_op
        self.add_flow(load_op, rec_op)
        
        self.add_flow(rec_op, converterReconstructed)
        
        self.add_flow(rec_op, denoiser)
        self.add_flow(denoiser, converterDenoised)


if __name__ == "__main__":
    app = cbct_reconstruction()
    app.run()
