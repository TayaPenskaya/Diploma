FROM python:3.7
MAINTAINER KOT

WORKDIR /usr/project

COPY . .

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 1488

CMD ["python","./init.py"]


