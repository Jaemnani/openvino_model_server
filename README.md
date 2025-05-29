# OpenVINO Model Server Example with Docker

## Docker 사용 준비
```
sudo apt update

sudo apt install git docker.io make cmake

sudo chmod 666 /var/run/docker.sock
```

## how to build Docker image of OpenVINO Model Server
* 만약 GPU 버전이 피룡하다면, ./models/config.json의 model config list에서 target device를 "GPU"로 변경해 준 후, 다시 빌드. (빌드되는 도커 이름에 주의할 것.)
```
./ovms_build.sh
```

## how to run docker image
```
./ovms_run.sh
```

## how to build custom node C++ source 
```
./make_custom_model.sh
```

## how to use client example
```

```

