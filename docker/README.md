
# Use the Dockerfile to build images for FedLab environment


- Step 1

Find an appropriate base pytorch image for your platform from dockerhub https://hub.docker.com/r/pytorch/pytorch/tags. Then, replace the value of TORCH_CONTAINER in demo dockerfile.

- Step 2

To install specific pytorch version, you need to choose a correct install command, which you can find in https://pytorch.org/get-started/previous-versions/. Then, modify the 16-th command in demo dockerfile.

- Step 3

Build the images for your own platform by running command below.
> docker build -t image_name .

- Note
  
If you are not in China area, it is ok to remove line 11,12 and "-i https://pypi.mirrors.ustc.edu.cn/simple/" in line 19.

**Run docker images with "--gpus all" to enable cuda for container and "--network=host" to enable the network communication.