# This is a example of fedlab installation via Dockerfile

# replace the value of TORCH_CONTAINER with pytorch image that satisfies your cuda version
# you can finde it in https://hub.docker.com/r/pytorch/pytorch/tags
ARG TORCH_CONTAINER=1.5-cuda10.1-cudnn7-runtime

FROM pytorch/pytorch:${TORCH_CONTAINER}

RUN pip install --upgrade pip \
    & pip uninstall -y torch torchvision  \
    & conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ \
    & conda config --set show_channel_urls yes \
    & mkdir /root/tmp/

# replace with the correct install command, which you can find in https://pytorch.org/get-started/previous-versions/
RUN conda install -y pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch 

# pip install fedlab
RUN TMPDIR=/root/tmp/ pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ fedlab

