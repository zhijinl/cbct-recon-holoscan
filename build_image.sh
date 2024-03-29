#! /bin/bash
## ---------------------------------------------------------------------------
##
## File: build-container.sh for project: CBCT Recon
##
## Created by Zhijin Li
## E-mail:   <zhijinl@nvidia.com>
##
## Started on  Fri Nov 24 12:01:57 2023 Zhijin Li
## Last update Fri Nov 24 12:03:02 2023 Zhijin Li
## ---------------------------------------------------------------------------


docker build - < ./docker/holoscan-v0.6-astra.dockerfile -t holoscan-sdk-astra:v0.6.0-dgpu
