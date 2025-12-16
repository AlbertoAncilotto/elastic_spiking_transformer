FROM nvcr.io/nvidia/pytorch:24.03-py3

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update

ARG USER=e3da
ARG USER_ID=1006 # uid from the previus step
ARG USER_GROUP=e3da
ARG USER_GROUP_ID=1006 # gid from the previus step
ARG USER_HOME=/home/${USER}


RUN apt install -y wget git python3 python3-venv libgl1 libglib2.0-0

RUN groupadd --gid $USER_GROUP_ID $USER \
    && useradd --uid $USER_ID --gid $USER_GROUP_ID -m $USER

USER $USER

RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install --no-dependencies micromind
WORKDIR $USER_HOME

