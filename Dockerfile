FROM python:3.8-slim-buster

WORKDIR /app

COPY . /app

ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}  #pass the values for this env variables at run time


RUN pip install -r requirements.txt

EXPOSE 5000

CMD [ "python3","app.py" ]