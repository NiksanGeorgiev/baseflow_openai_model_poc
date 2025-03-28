# Use an official Python runtime as a parent image.
FROM python:latest

# Set the working directory in the container.
WORKDIR /app

RUN git clone https://github.com/NiksanGeorgiev/baseflow_openai_model_poc.git /tmp/repo && \
    mkdir -p /app && \
    cp -R /tmp/repo/. /app && \
    rm -rf /tmp/repo


# Expose port 8080 for the Flask app.
EXPOSE 8080

# Install any needed packages specified in requirements.txt.
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Run the application.
CMD ["python", "embed_search.py"]
