FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create a directory for summaries
RUN mkdir -p /app/summaries

# Expose the port for the web viewer
EXPOSE 12000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the web viewer by default
CMD ["python", "web_viewer.py", "--host", "0.0.0.0", "--port", "12000", "--summaries_dir", "summaries"]