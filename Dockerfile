FROM python:3.10-alpine

ADD main.py .

RUN pip install -r requirements.txt

CMD ["python", "./main.py"]