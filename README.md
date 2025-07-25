# live-pose

Intel RealSense Depth Camera compatible Python package for live 6 DOF pose estimation.

Forked from https://github.com/Kaivalya192/live-pose

## Table of Contents

- [Installation](#installation)
- [Preparation](#preparation)
- [Usage](#usage)

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/HealorCai/live-pose.git
    cd live-pose
    ```

## Preparation

### Docker Build

1. **Build the Docker container**:
    ```sh
    cd docker
    docker build --network host -t foundationpose .
    ```

2. **Install Weights**:
   - Download the weights from [this link](https://drive.google.com/drive/folders/1wJayPZzZLZb6sxm6EeOQCJvzOAibJ693?usp=sharing) and place them under `live-pose/FoundationPose/weights`.

## Usage

### Running the Container

1. **Run the container**:
    ```sh
    bash docker/run_container.sh
    ```
    note: To run on windows install `Cygwin` and execute `./docker/run_container_win.sh`
### Building packages

1. **Build**:
    ```sh
    CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build.bash
    ```
### Running the Model

1. **Run the live pose estimation**:
   Clone the repo of [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) and check the **TODO** in `FoundationPose/get_6dpose.py`
   ```sh
    bash get_6dpose.sh
    ```
    note: To run on windows install `Cygwin` and execute `./get_6dpose.sh`
   
2. **Locate the .obj file**:
    <br> Note: For novel object you can use [Object Recustruction Framework](https://github.com/Kaivalya192/Object_Reconstruction) </br>
    
