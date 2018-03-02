FROM jingpengw/chunkflow.jl:master

LABEL   maintainer="Jingpeng Wu"\
        project="convnet"

RUN apt install -y -qq git python-pip libssl-dev libffi-dev
RUN pip install --upgrade pip
RUN pip install cython

WORKDIR /opt 
RUN git clone https://github.com/jingpengw/patchprovider.git 
WORKDIR patchprovider
RUN pip install -r requirements_dev.txt
RUN make install

WORKDIR ~  
ENTRYPOINT /bin/bash
