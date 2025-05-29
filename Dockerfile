FROM openvino/model_server:latest-gpu
USER root
ENV LD_LIBRARY_PATH=/ovms/lib
ENV PYTHONPATH=/ovms/lib/python
RUN apt update && apt install -y python3-pip git
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN pip3 install numpy==1.26.4 opencv-python-headless==4.11.0.86
RUN pip3 install tritonclient[grpc]==2.54.0
USER ovms
ENTRYPOINT [ "/ovms/bin/ovms" ]
COPY ./models/ /models/