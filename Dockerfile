FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt
COPY api ./api
COPY app ./app
COPY vision ./vision
COPY solver ./solver
COPY start.sh .
RUN chmod +x start.sh
ENV API_URL=http://127.0.0.1:8000
EXPOSE 7860
CMD ["./start.sh"]