#!/bin/bash

# Script to run homography test
# Usage: ./run_homography_test.sh <path_to_image> <output_width> <output_height>
IMAGE_PATH=""
OUTPUT_WIDTH=0
OUTPUT_HEIGHT=0

python ./homography_test.py "$IMAGE_PATH" $OUTPUT_WIDTH $OUTPUT_HEIGHT
