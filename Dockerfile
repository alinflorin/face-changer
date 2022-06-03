FROM python
WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=boot
ENV FLASK_ENV=production
CMD [ "python", "./boot.py" ]
EXPOSE 5000