#FROM tensorflow/tensorflow:latest-gpu-jupyter
FROM tensorflow/tensorflow:2.4.2-gpu-jupyter

SHELL [ "/bin/bash", "-l", "-c" ]

USER root
ENV NVM_DIR /usr/local/nvm
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

RUN pip install --upgrade pip
RUN pip install jupyter_contrib_nbextensions version_information jupyterlab  && \
    jupyter contrib nbextension install --sys-prefix

#RUN conda update conda

# Add requirements file
COPY jupyter_notebook_config.py /etc/jupyter/
COPY requirements.txt /app/
RUN pip install wheel -r /app/requirements.txt
RUN mkdir /opt/app
WORKDIR /

# Jupyter with Docker Compose
EXPOSE 8888


