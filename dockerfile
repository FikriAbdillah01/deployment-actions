FROM python:3.11-slim
RUN pip install -r requirements.txt

USER nobody
RUN pip isntall -r requirements.txt

ENTRYPOINT [ "python", "app.py" ]