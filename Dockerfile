# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir opencv-python-headless dlib numpy scikit-learn

# Download and extract the required model files
RUN mkdir -p /root/Desktop/opencv_data && \
    wget https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2 -O /root/Desktop/opencv_data/shape_predictor_68_face_landmarks.dat.bz2 && \
    bunzip2 /root/Desktop/opencv_data/shape_predictor_68_face_landmarks.dat.bz2 && \
    wget https://github.com/davisking/dlib-models/raw/master/dlib_face_recognition_resnet_model_v1.dat.bz2 -O /root/Desktop/opencv_data/dlib_face_recognition_resnet_model_v1.dat.bz2 && \
    bunzip2 /root/Desktop/opencv_data/dlib_face_recognition_resnet_model_v1.dat.bz2

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME FaceRecognitionApp

# Run app.py when the container launches
CMD ["python", "recognition.py"]
