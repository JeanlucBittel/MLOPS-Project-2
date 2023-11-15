FROM python:3.9-alpine

# Add all files
ADD main.py .
ADD setup.py .
ADD requirements.txt .

# Add all firectories
ADD src .
