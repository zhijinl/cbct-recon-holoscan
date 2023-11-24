#! /bin/bash
## ---------------------------------------------------------------------------
##
## File: launch_docker.sh for project: CBCT Recon
##
## Created by Zhijin Li
## E-mail:   <zhijinl@nvidia.com>
##
## Started on  Fri Nov 24 11:07:21 2023 Zhijin Li
## Last update Fri Nov 24 13:54:59 2023 Zhijin Li
## ---------------------------------------------------------------------------


export NGC_CONTAINER_IMAGE_PATH="holoscan-sdk-astra:v0.6.0-dgpu"

xhost +local:docker

# Find the nvidia_icd.json file which could reside at different paths
# Needed due to https://github.com/NVIDIA/nvidia-container-toolkit/issues/16
nvidia_icd_json=$(find /usr/share /etc -path '*/vulkan/icd.d/nvidia_icd.json' -type f,l -print -quit 2>/dev/null | grep .) || (echo "nvidia_icd.json not found" >&2 && false)

# --ipc=host, --cap-add=CAP_SYS_PTRACE, --ulimit memlock=-1 are needed for the distributed applications using UCX to work.
# See https://openucx.readthedocs.io/en/master/running.html#running-in-docker-containers

# Run the container
docker run -it --rm --net host \
  --runtime=nvidia \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $nvidia_icd_json:$nvidia_icd_json:ro \
  -v <YOUR WORKSPACE>:/cbct-recon \
  -w /cbct-recon \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display \
  -e DISPLAY=$DISPLAY \
  --ipc=host \
  --cap-add=CAP_SYS_PTRACE \
  --ulimit memlock=-1 \
  ${NGC_CONTAINER_IMAGE_PATH}
