#!/usr/bin/env bash

# Absolute path to this script; resolves symbolic links
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
# get the directory of the script
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

# check if $SCRIPT_DIR/build is a directory and exists
if [ -d "$SCRIPT_DIR/build" ]; then
    echo "Build directory exists, removing it..."
    rm -r "$SCRIPT_DIR/build"
fi

# create build directory
mkdir "$SCRIPT_DIR/build"
# build the project
cd "$SCRIPT_DIR/build" && cmake -DCMAKE_INSTALL_PREFIX=/opt/ros/humble .. && make -j$(nproc) && make install