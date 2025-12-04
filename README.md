## 📝 OpenVINO Model Server Example (Docker Environment)

### 🚀 Project Overview

This project provides a Docker-based example for serving deep learning model inference using **OpenVINO Model Server (OVMS)**. Specifically, it utilizes the **Custom Node** feature to configure **preprocessing and postprocessing pipelines** within OVMS for **DeepLabV3 (Segmentation)** and **YOLOX-Tiny (Object Detection)** models. 

---

### 🐳 Docker Environment Setup

#### 1. Prepare for Docker Usage

Use the commands below to install necessary packages and configure Docker permissions.

```bash
sudo apt update
sudo apt install git docker.io make cmake
# Grant current user permission to run docker commands (re-login required)
sudo usermod -aG docker $USER 
# Change permissions for docker.sock (Temporary measure, using usermod above is recommended)
sudo chmod 666 /var/run/docker.sock
```

#### 2. Build OpenVINO Model Server Docker Image

Use the `ovms_build.sh` script to build the image. By default, the **CPU version** with the tag `ovms_cpu` is used.

```bash
./ovms_build.sh
```

> **💡 GPU Version Note:** If you require the **GPU version**, you must change the `target_device` to **`"GPU"`** for the relevant model in the `model_config_list` within `./models/config.json` before building.

#### 3. Run the Docker Image

The `ovms_run.sh` script runs the model server in the background. This script includes permissions to access the **GPU device (`/dev/dri`)**, and the server runs on **port 9000**.

```bash
./ovms_run.sh
```
### ⚙️ Custom Node and Pipeline Information

The server provides two main model inference pipelines. The preprocessing and postprocessing logic using **Custom Nodes** is integrated into these pipelines. 

| Pipeline Name | Model | Task | Custom Node Usage |
| :--- | :--- | :--- | :--- |
| **`custom_deeplabv3`** | `deeplabv3` | Semantic Segmentation | `deeplabv3_preprocessing`, `deeplabv3_postprocessing` |
| **`custom_yolox`** | `yolox_tiny` | Object Detection | `yolox_preprocessing`, `yolox_postprocessing` |

#### 1. Build Custom Node C++ Source

Build the C++ source code for the Custom Nodes to generate dynamic libraries (`.so` files) and copy them to the models directory.

```bash
./make_custom_model.sh
```
### 📞 Client Usage Example

Clients can send inference requests to the server using **gRPC** or **REST API**.

#### 1. Using a Python gRPC Client (Example)

OpenVINO Model Server adheres to the **Triton Inference Protocol**, so libraries like `tritonclient` can be used to send requests. 

**Key Request Information:**

| Pipeline | Target Endpoint (gRPC) | Input Name (Input Data Item) | Output Name (Output Data Item) |
| :--- | :--- | :--- | :--- |
| **`custom_yolox`** | `custom_yolox` | `data` | `detect_out` |
| **`custom_deeplabv3`** | `custom_deeplabv3` | `data` | `segment_out` |

**Client Execution (Placeholder)**

```bash
# Execute from the directory containing the client code (example)
# First, ensure the Triton Client library is installed:
# python3 -m pip install tritonclient[grpc]

# Example execution for the YOLOX detection pipeline:
python3 client_script.py --pipeline_name custom_yolox --image_path /path/to/image.jpg
```
