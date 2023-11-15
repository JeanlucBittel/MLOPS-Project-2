FROM bitnami/pytorch:latest

# Set working directory
WORKDIR /mlops/

# Copy the necessary files
COPY main.py /mlops/main.py
COPY setup.py /mlops/setup.py
COPY requirements.txt /mlops/requirements.txt

# Add the necessary directories
ADD src /mlops/src

# Run all commands
RUN python3 -m pip install -r /mlops/requirements.txt
RUN mkdir -p /mlops/.cache

CMD ["python3", "/mlops/main.py"]