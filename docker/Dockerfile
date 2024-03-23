FROM python:3.12-alpine

COPY ./app/requirements.txt /requirements.txt

# Run with no cache
RUN pip install --no-cache-dir --upgrade -r /requirements.txt

COPY ./app /app

WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
