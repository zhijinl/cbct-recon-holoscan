export MONAI_LABEL_DICOMWEB_USERNAME=orthanc
export MONAI_LABEL_DICOMWEB_PASSWORD=orthanc
export MONAI_LABEL_DICOMWEB_CONVERT_TO_NIFTI=false

#monailabel start_server --app apps/radiology --studies http://127.0.0.1:8042/dicom-web --conf models segmentation_spleen
# --port 8002
monailabel start_server --app apps/monaibundle --studies http://127.0.0.1:8042/dicom-web --conf models wholeBody_ct_segmentation 
