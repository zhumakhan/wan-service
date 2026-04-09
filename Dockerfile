FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime
ARG GIT_TAG=unknown
ENV GIT_TAG=${GIT_TAG}
ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=dumb
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Add deadsnakes PPA to get Python 3.11
RUN apt update && apt install -y --no-install-recommends \
    python3-pip \
    git wget nano vim ffmpeg curl \
    build-essential unzip ncdu && \
    rm -rf /var/lib/apt/lists/*


RUN pip3 install packaging --break-system-packages
COPY requirements.txt requirements.txt 
RUN pip3 install --no-cache-dir -r requirements.txt --break-system-packages

ENV OMP_NUM_THREADS=8

RUN pip3 install scikit-image --break-system-packages

EXPOSE 8000

COPY . .