FROM python:3.10-slim

# Update distro
RUN apt update
RUN apt upgrade -y

# Helper tool for querying installed libs
RUN apt install -y pkg-config

# Install Tesseract
RUN apt install -y tesseract-ocr

# Install pip packages
RUN pip install numpy
RUN pip install pytesseract
RUN pip install matplotlib

# OpenCV

# RUN pip install opencv-contrib-python-headless

# Deps
RUN apt install -y cmake g++ wget unzip
RUN apt install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libx264-dev
# Download and unpack sources
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
RUN wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
RUN unzip opencv.zip
RUN unzip opencv_contrib.zip
# Create build dir
RUN mkdir -p build
RUN cd build
# Configure
RUN cmake -DWITH_FFMPEG=1 -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x
# Build
RUN make -j32
RUN make install

WORKDIR /var/src

# Keep container open
ENTRYPOINT tail -f /dev/null