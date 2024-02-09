import numpy as np
import pydicom
import os
import subprocess

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from pydicom.valuerep import PersonName
from pydicom.uid import generate_uid
from PARAM import *
from datetime import datetime

class ConvertOp(Operator):
    def __init__(self, *args, **kwargs):
        self.dicom_dir_path = os.path.join(PATH,"output/")
        self.files = []
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in_convert")
        spec.param("patientName")

    def upload_dicom_files(self):
        command = ['storescu', '-aec', 'ORTHANC', 'localhost', '4242'] + self.files
        result = subprocess.run(command)
        if result.returncode == 0:
            print("DICOM files uploaded successfully.")
        else:
            print("Failed to upload DICOM files.")
        
    def compute(self, op_input, op_output, context):
        self.start_time = datetime.now()
        reconstructed = self.patientName == "Reconstructed"

        output = self.dicom_dir_path + ("dicom_slices_reconstructed/" if reconstructed else "dicom_slices_denoised/")
        
        if reconstructed:
            print()
        print("Upload")
        
        os.system("rm " + output+"*.dcm")
        os.system("cp "+ os.path.join(PATH,"data/","dicom_templates/") + "*.dcm"+" "+ output)
        
        self.files = [os.path.join(output, f) for f in sorted(os.listdir(output)) if f.endswith('.dcm')]
        
        npy_array = op_input.receive("in_convert")
        i=0
        newUID = generate_uid()
        # Loop through all files in the directory
        for filename in self.files:
            # Check if the file is a DICOM file
            if filename.lower().endswith('.dcm'):
                # Construct the full file path
                file_path = os.path.join(output, filename)
                
                # Read the DICOM file
                ds = pydicom.dcmread(file_path)
                ds.PatientName = self.patientName
                ds.StudyInstanceUID = newUID
                
                # Ensure the pixel data is in a modifiable format
                pixels = ds.pixel_array
                slice_2d = npy_array[i, :, :]
                
                my_max = np.max(slice_2d)
                my_min = np.min(slice_2d)
        
                pixels[:slice_2d.shape[0], :slice_2d.shape[1]] = ((slice_2d - my_min) / (my_max - my_min) * 5000).astype(int)

                new_pixels = pixels[:slice_2d.shape[0], :slice_2d.shape[1]]
                ds.Rows, ds.Columns = new_pixels.shape   
                ds.PixelData = new_pixels.tobytes()
                
                # Save the modified DICOM file in the new directory
                ds.save_as(file_path)
                i+=1
                
        self.upload_dicom_files()
        
        print(f"Upload Time:",datetime.now() - self.start_time)
        if reconstructed:
            print()
        
