FROM bitnami/pytorch:latest

# Add all files
COPY main.py /mlops/main.py
COPY setup.py /mlops/setup.py
COPY requirements.txt /mlops/requirements.txt

# Add all directories
ADD src /mlops/src

RUN python3 -m pip install -r /mlops/requirements.txt

WORKDIR /mlops

CMD ["python3", "/mlops/main.py"]