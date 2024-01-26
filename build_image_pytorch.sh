#! /bin/bash
## ---------------------------------------------------------------------------
##
## File: build-container_pytorch.sh for project: CBCT Recon
##
## Created by Zhijin Li
## E-mail:   <zhijinl@nvidia.com>   
## Modified by Teo Le Bihan
## E-mail:   <teo.le-bihan@epita.fr>
##
## Started on  Fri Nov 24 12:01:57 2023 Zhijin Li
## Last update Fri Jan 26 23:53:00 2024 Teo Le Bihan
## ---------------------------------------------------------------------------


docker build - < ./docker/pytorch-v22.03-holoscan-v0.6-astra.dockerfile -t pytorch-holoscan-sdk-astra:v0.6.0-dgpu
