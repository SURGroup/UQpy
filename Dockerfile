#RUN curl -fsSL https://get.docker.com -o get-docker.sh
#RUN chmod +x get-docker.sh
#RUN sh get-docker.sh

# Build the image based on the official Python version 3.9 image
FROM python:3.9

# Use RUN to install Python packages (numpy and scipy) via pip, Python's package manager
RUN pip3 install UQpy