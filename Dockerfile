# Dockerfile

# Use the official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app
COPY . .

# Expose the port Streamlit will run on
EXPOSE 8080

# Disable Streamlit file watcher to avoid torch.classes error (MAY DELETE IT)
# ENV STREAMLIT_WATCHER_TYPE=none 

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
