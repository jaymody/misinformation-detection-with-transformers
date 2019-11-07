FROM pytorch/pytorch:latest

ADD . /usr/src/
WORKDIR /usr/src/

RUN pip3 install -r requirements

CMD [ "bash", "/usr/src/run" ]
