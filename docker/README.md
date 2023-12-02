
# Use the Dockerfile to build images for FedLab environment


- Step 1

Find an appropriate base PyTorch image for your platform from dockerhub https://hub.docker.com/r/pytorch/pytorch/tags. Then, replace the value of TORCH_CONTAINER in demo dockerfile.

- Step 2

To install specific PyTorch version, you need to choose a correct install command, which can be found at https://pytorch.org/get-started/previous-versions/. Then, modify the 16-th command in demo dockerfile.

- Step 3

Build the images for your own platform by running command below.
> $ docker build -t image_name .

- Note
  
Please be sure to use "--gpus all" and "--network=host" when starting a docker container:

> $ docker run -itd --gpus all --network=host b23a9c46cd04(image name) /bin/bash

If you are not in China area, it is ok to remove line 11, 12 and "-i https://pypi.mirrors.ustc.edu.cn/simple/" in line 19.
