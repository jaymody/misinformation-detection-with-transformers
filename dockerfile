FROM nvidia/cuda:10.0-base-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=3.7 && \
     /opt/conda/bin/conda clean -afy && \
     ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
     echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
     echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH
RUN /bin/bash -c "source activate base"

WORKDIR /usr/src/
ADD . /usr/src

RUN conda install pytorch cudatoolkit=10.0 -c pytorch && \
    conda install scikit-learn tqdm nltk pandas && \
    conda clean -afy

RUN pip install tensorboardx simpletransformers
RUN python -m nltk.downloader punkt

CMD [ "bash", "/usr/src/run" ]
