FROM python:3.11-slim

COPY requirements.txt api/requirements.txt

WORKDIR api/

RUN pip install -r requirements.txt

COPY . /api

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]