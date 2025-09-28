FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY src/ ./src/
COPY main_himas_federated.py .
COPY hospital_client.py .

# Expose port for Flower server
EXPOSE 8080

# Default command runs the federated server
CMD ["python", "-c", "from src.federated.flower_server import start_federated_server; start_federated_server()"]