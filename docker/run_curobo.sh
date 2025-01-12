#!/bin/bash
set -ex
USERNAME="ros"
docker run -v $PWD/../src/:/home/${USERNAME}/ros2_ws/src \
	-v $PWD/../cudampl/:/home/${USERNAME}/cudampl \
	-e DISPLAY \
	-e QT_X11_NO_MITSHM=1 \
	-e XAUTHORITY \
	-e NVIDIA_DRIVER_CAPABILITIES=all \
	--ipc=host \
	--gpus all \
	--network="host" \
	-p 8888:8888 \
	-p 6006:6006 \
	--privileged=true \
	-v /etc/localtime:/etc/localtime:ro \
	-v /dev/video0:/dev/video0 \
	-v "/tmp/.X11-unix:/tmp/.X11-unix:rw" -p 19997:19997 --rm -it isaac-sim-ros2-curobo bash