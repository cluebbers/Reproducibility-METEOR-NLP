FROM continuumio/miniconda3

RUN apt-get update && apt-get -y install gcc g++ build-essential less
RUN conda install -y pip ipython

ENV PYTHONDONTWRITEBYTECODE 1
ENV TZ America/New_York

RUN mkdir -p /usr/share/man/man1
RUN apt-get update && \
   DEBIAN_FRONTEND=noninteractive \
   apt-get -y install default-jre-headless && \
   apt-get clean && \
   rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
   apt-get install -y libxml-parser-perl

RUN pip install \
    beautifulsoup4 \
    datasets \
    dill \
    flask \
    lxml \
    matplotlib \
    nltk \
    numpy \
    pandas \
    requests \
    scipy \
    tqdm \
    psutil

RUN pip install \
    mypy \
    pytest \
    pytest-mypy \
    pytest-watch 

RUN python -c "\
from datasets import load_dataset;\
print(load_dataset('cnn_dailymail', '3.0.0')['train'][0]);"

RUN python -c "\
import nltk;\
nltk.download('punkt');"