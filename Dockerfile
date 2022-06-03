FROM python
WORKDIR /usr/src/app

RUN sudo apt install cmake

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=boot
ENV FLASK_ENV=production
CMD [ "python", "./boot.py" ]
EXPOSE 5000