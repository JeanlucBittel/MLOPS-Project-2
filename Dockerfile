FROM bitnami/pytorch:latest

# Add all files
COPY main.py .
COPY setup.py .
COPY requirements.txt .

# Add all directories
ADD src .

RUN python3 -m pip install -r requirements.txt

CMD ["python3", "main.py"]