FROM python
WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1 \
    libopenblas-dev \
    python3-opencv \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt -vvv

COPY . .

ENV FLASK_APP=boot
ENV FLASK_ENV=production
CMD [ "python", "./boot.py" ]
EXPOSE 5000