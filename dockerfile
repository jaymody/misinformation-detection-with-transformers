FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

RUN apt update && \
    apt install -y bash \
        build-essential \
        git \
        curl \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        python3 \
        python3-pip && \
    rm -rf /var/lib/apt/lists

WORKDIR /usr/src/
COPY . /usr/src/

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt && \
    python3 -m pip install --no-cache-dir . && \
    python3 -m nltk.downloader punkt && \
    python3 -m spacy download en_core_web_lg

CMD [ "bash", "/usr/src/run" ]
