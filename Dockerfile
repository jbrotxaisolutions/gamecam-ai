# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
# This is done first to leverage Docker's layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Command to run the application using Gunicorn
# -w 4: Use 4 worker processes
# -b 0.0.0.0:5001: Bind to all network interfaces on port 5001
# app:app: Look for the 'app' variable in the 'app.py' file
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "app:app"]