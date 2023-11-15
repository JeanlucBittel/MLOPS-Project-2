FROM pytorch/pytorch:latest

# Add all files
ADD main.py ./mlops/main.py
ADD setup.py ./mlops/setup.py

COPY ./requirements.txt ./mlops/requirements.txt

# Add all directories
ADD src ./mlops/src

RUN pip install -r ./mlops/requirements.txt