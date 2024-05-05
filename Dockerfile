FROM ubuntu:24.04

RUN apt-get update --fix-missing && \
    apt-get upgrade -y && \
    apt-get install -yq wget bzip2 ca-certificates curl git nano libgl1 libgl1-mesa-dev && \
    apt-get clean && apt-get check && apt-get autoclean && apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /usr/src/*

RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    bash Miniforge3-$(uname)-$(uname -m).sh -b -p /opt/conda && \
    rm Miniforge3-$(uname)-$(uname -m).sh && \
    /opt/conda/bin/conda init bash

RUN mkdir -p /src/P2_pipeline
COPY ./src/ /src/

RUN /opt/conda/bin/mamba create -n p2 -y python=3.10 pip ipykernel pytorch torchvision torchaudio pytorch-cuda=12.1 cuda-libraries-dev cuda-profiler-api cuda-compiler openslide openslide-python pyvips albumentations kornia torchmetrics onnx segmentation-models-pytorch timm lightning lovely-tensors lovely-numpy torchinfo nvidia-ml-py matplotlib seaborn scienceplots tqdm loguru python-dotenv icecream py-cpuinfo hjson-py pydantic dataclasses cloudpickle pyarrow polars tensorboardx ipywidgets gdown shapely lifelines -c pytorch -c nvidia -c conda-forge
RUN /opt/conda/envs/p2/bin/pip install bitsandbytes deepspeed focal_loss_torch patchify git+https://github.com/nik-shvetsov/torch-staintools --no-deps

RUN /opt/conda/bin/conda clean --all -yq

RUN /opt/conda/envs/p2/bin/pip install -e /src/src/roi_segment/
RUN /opt/conda/envs/p2/bin/pip install -e /src/src/patch_class/
RUN /opt/conda/envs/p2/bin/pip install -e /src/src/hover_quant/

WORKDIR /src/

CMD [ "/bin/bash" ]
