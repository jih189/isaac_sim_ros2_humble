#!/bin/bash
IMAGE_NAME="isaac-sim-ros2-curobo"
container_count=$(docker ps | grep $IMAGE_NAME | wc -l)
if [ $container_count -eq 1 ]; then
        last_container_id=$(docker container ls  --filter=ancestor=$IMAGE_NAME --format "{{.ID}}")
        docker exec -it $last_container_id /bin/bash
else
        echo "you have $container_count container with image $IMAGE_NAME, you should have only one!!"
fi