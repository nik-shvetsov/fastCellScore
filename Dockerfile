FROM ubuntu:24.04 as builder

RUN apt-get update --fix-missing && \
    apt-get install -yq --no-install-recommends wget bzip2 ca-certificates curl git

RUN curl -L -o Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    bash Miniforge3.sh -b -p /opt/conda && \
    rm Miniforge3.sh && \
    /opt/conda/bin/conda install -c conda-forge mamba -y && \
    /opt/conda/bin/mamba create -n default --copy -y python=3.10 pip ipykernel pytorch torchvision torchaudio pytorch-cuda=12.1 cuda-libraries-dev cuda-profiler-api cuda-compiler openslide openslide-python pyvips albumentations kornia torchmetrics onnx segmentation-models-pytorch timm lightning lovely-tensors lovely-numpy torchinfo nvidia-ml-py matplotlib seaborn scienceplots tqdm loguru python-dotenv icecream py-cpuinfo hjson-py pydantic dataclasses cloudpickle pyarrow polars tensorboardx ipywidgets gdown shapely lifelines -c pytorch -c nvidia -c conda-forge && \
    /opt/conda/bin/conda clean --all --force-pkgs-dirs -y

COPY ./src/ /src/

RUN /opt/conda/envs/default/bin/pip --no-cache-dir install bitsandbytes deepspeed focal_loss_torch patchify git+https://github.com/nik-shvetsov/torch-staintools --no-deps && \
    /opt/conda/envs/default/bin/pip --no-cache-dir install -e /src/src/roi_segment/ && \
    /opt/conda/envs/default/bin/pip --no-cache-dir install -e /src/src/patch_class/ && \
    /opt/conda/envs/default/bin/pip --no-cache-dir install -e /src/src/hover_quant/

FROM ubuntu:24.04

RUN apt-get update --fix-missing && \
    apt-get upgrade -yq && \
    apt-get install -yq --no-install-recommends libgl1 libegl1 libopengl0 && \
    apt-get clean && apt-get check && apt-get autoclean && apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /src /src

ENV PATH=/opt/conda/envs/default/bin:$PATH
ENV CONDA_DEFAULT_ENV=default
ENV CONDA_PREFIX=/opt/conda/envs/default
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

WORKDIR /src/

CMD [ "/bin/bash" ]
