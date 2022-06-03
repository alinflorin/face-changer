FROM python:alpine
WORKDIR /usr/src/app
RUN apk --no-cache add musl-dev linux-headers g++
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV FLASK_APP=boot
ENV FLASK_ENV=production
CMD [ "python", "./boot.py" ]
EXPOSE 5000