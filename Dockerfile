FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Add all files
ADD main.py ./mlops/main.py
ADD setup.py ./mlops/setup.py

COPY ./requirements.txt ./mlops/requirements.txt

# Add all directories
ADD src ./mlops/src

RUN pip install -r ./mlops/requirements.txt