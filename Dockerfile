FROM python:3.7.3-slim

RUN pip install tensorflow==2.0.0-alpha0 matplotlib pillow scipy requests flask flask-restful jsonify

WORKDIR /app

COPY models/ models/
COPY server.py server.py
COPY Pipfile Pipfile


EXPOSE 8080

ENTRYPOINT [ "python", "./server.py" ]

