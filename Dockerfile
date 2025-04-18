
FROM python:latest

# Install system dependencies if needed
RUN apt-get update && apt-get install -y libpq-dev gcc

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY . .

# Create a directory for persistent data
RUN mkdir -p /app/data

# Expose the port
EXPOSE 8080

CMD ["python", "app.py"]
