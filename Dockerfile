WORKDIR /app

RUN sudo apt-get update
RUN sudo apt-get install libgl1-mesa-glx -y

