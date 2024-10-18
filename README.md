# LiDAR Point Cloud Object Tracking

## Overview

This project focuses on tracking objects in LiDAR point cloud data using boundary boxes. The tracking is performed by measuring distances between frames, predicting future positions with a Kalman filter, and comparing velocities to improve tracking accuracy. Additionally, it provides a visualization tool to generate a bird's eye view (BEV) of the tracked objects as a video or GIF.

### Input Format
The input to the tracking system is a list of boundary boxes, each defined by 7 parameters:
- **3 values** for the center of the box (`x`, `y`, `z`)
- **3 values** for the extent of the box (`x`, `y`, `z`)
- **1 value** for the yaw angle (rotation)

### Output
1. **Object Tracking**: The tracking system outputs a `3d_ann.json` file that contains the boundary boxes and their associated tracking IDs.
2. **Bird's Eye View (BEV) Visualization**: This tool generates a BEV video or GIF of the object tracking results using the LiDAR data.

## Key Features
### 1. Object Tracking:
- **2D Space Tracking**: Tracks objects based on the closest distance in the X, Y 2D space. The `z` (height) is ignored for tracking.
- **Kalman Filter**: A Kalman filter is used to predict the next position of a boundary box, enhancing tracking accuracy.
- **Velocity Comparison**: Velocity comparison between consecutive frames is used to improve the object association and reduce ID switching errors.

### 2. Bird's Eye View Visualization:
- **BEV Video/GIF**: Generates a birds-eye view representation of the object tracking results, allowing for visual inspection.

## Installation

Install the required dependencies by running the following commands:

```bash
pip install -r requirements.txt
pip install json
pip install re
