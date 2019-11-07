FROM pytorch/pytorch:latest

ADD valerie/ /usr/src/
ADD model/ /usr/src/
ADD requirements.txt /usr/src/
WORKDIR /usr/src/

RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt

CMD [ "bash", "/usr/src/run" ]
