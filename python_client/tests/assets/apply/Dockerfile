FROM ghcr.io/run-house/server:v3
RUN pip install requests
COPY test_data.txt /app/test_data.txt
COPY run.py /app/run.py
CMD ["python", "/app/run.py"]
