FROM python:3.12.6

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

COPY ./.env /code/.env

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir sentence-transformers

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./assets/ /code/assets

COPY ./app/ /code/app

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]