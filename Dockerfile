FROM python:3.10.13

COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN mkdir /data
RUN wget https://storage.yandexcloud.net/eremeev-d-bucket-main/1722760266.tar -O data.tar
RUN tar -xf data.tar -C /data
RUN rm data.tar

RUN mkdir /app
COPY app app

EXPOSE 8501

ENV PYTHONPATH .

ENTRYPOINT ["streamlit", "run", "app/main.py", "--server.port=8501"]