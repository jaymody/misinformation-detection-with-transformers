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

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
        mkl \
        torch \
        scikit-learn \
        scipy \
        spacy \
        wordninja \
        tldextract \
        tqdm \
        pandas \
        nltk \
        bs4 \
        requests \
        gensim && \
    python3 -m pip install --no-cache-dir git+https://github.com/huggingface/transformers && \
    python3 -m pip install --no-cache-dir . && \
    python3 -m nltk.downloader punkt && \
    python -m spacy download en_core_web_lg

WORKDIR /usr/src/
COPY . /usr/src/

CMD ["/bin/bash/run"]
