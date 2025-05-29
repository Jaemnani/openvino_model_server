//*****************************************************************************
// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include <iostream>
#include <map>
#include <string>

#include "../../custom_node_interface.h"
#include "../common/opencv_utils.hpp"
#include "../common/utils.hpp"
#include "opencv2/opencv.hpp"

static constexpr const char* TENSOR_NAME = "image";

int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    return 0;
}

int deinitialize(void* customNodeLibraryInternalManager) {
    return 0;
}

int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    // Parameters reading
    int _sourceImageHeight = get_int_parameter("input_h", params, paramsCount, -1);
    int _sourceImageWidth = get_int_parameter("input_w", params, paramsCount, -1);
    NODE_ASSERT(_sourceImageHeight > 0 || _sourceImageHeight == -1, "Source image height - when specified, must be larger than 0");
    NODE_ASSERT(_sourceImageWidth > 0 || _sourceImageHeight == -1, "Source image width - when specified, must be larger than 0");

    int _numClass = get_int_parameter("num_class", params, paramsCount, -1);
    NODE_ASSERT(_numClass > 0, "Number of class - must be larger than 0");

    // // Debug flag for additional logging.
    bool debugMode = get_string_parameter("debug", params, paramsCount) == "true";

    // // ------------ validation start -------------
    NODE_ASSERT(inputsCount == 1, "there must be exactly one input");
    const CustomNodeTensor* imageTensor = inputs;
    
    if(debugMode){
        std::cout << "Validation Start, Input Data Checking" << std::endl;
        std::cout << "inputsCount : " << inputsCount << std::endl;
        std::cout << "Tensor name : " << imageTensor->name << ";" << TENSOR_NAME << std::endl;
        std::cout << "dims count : " << imageTensor->dimsCount << std::endl;
    }
    
    NODE_ASSERT(std::strcmp(imageTensor->name, TENSOR_NAME) == 0, "node input name is wrong");
    NODE_ASSERT(imageTensor->dimsCount == 4, "image tensor shape must have 4 dimensions")
    NODE_ASSERT(imageTensor->dims[0] == 1, "image tensor must have batch size equal to 1")


    if (debugMode) {
        std::cout << "source image height : " << _sourceImageHeight    << std::endl;
        std::cout << "source image height : " << _sourceImageWidth     << std::endl;
        std::cout << "number of class : "     << _numClass             << std::endl;
        std::cout << "input shape[0] "      << imageTensor->dims[0] << std::endl;
        std::cout << "input shape[1] "      << imageTensor->dims[1] << std::endl;
        std::cout << "input shape[2] "      << imageTensor->dims[2] << std::endl;
        std::cout << "input shape[3]"      << imageTensor->dims[3] << std::endl;
    }
    // // ------------- validation end ---------------

    const float* output_buffer = (float*)imageTensor->data;

    const int height = 513;
    const int width = 513;

    // std::vector<uint8_t> argmax_result(height * width);
    uint64_t byteSize = sizeof(uint8_t) * height * width;
    // std::cout << "argmax_result size : " << argmax_result.size() << std::endl;
    // std::cout << "calcaulat bytesize : " << byteSize << std::endl;

    uint8_t* buffer = (uint8_t*)malloc(byteSize);
    NODE_ASSERT(buffer != nullptr, "malloc has failed.");

    for(int h = 0; h < height; ++h){
        for(int w =0; w < width; ++w){
            int index = h * width + w;
            float max_val = output_buffer[index];
            int max_index = 0;

            for (int c=1; c< _numClass; ++c) {
                float val = output_buffer[c * height * width + index];
                if (val > max_val){
                    max_val = val;
                    max_index = c;
                }
            }

            // argmax_result[index] = max_index*100; // for debug
            // argmax_result[index] = max_index;
            buffer[index] = max_index;
        }
    }

    *outputsCount = 1;
    *outputs = (struct CustomNodeTensor*)malloc(*outputsCount * sizeof(CustomNodeTensor));

    if ((*outputs) == nullptr) {
        std::cout << "malloc has failed" << std::endl;
        free(buffer);
        return 1;
    }

    CustomNodeTensor& output = (*outputs)[0];
    output.name = TENSOR_NAME;
    output.data = reinterpret_cast<uint8_t*>(buffer);
    output.dataBytes = byteSize;
    output.dimsCount = 2;
    output.dims = (uint64_t*)malloc(output.dimsCount * sizeof(uint64_t));
    NODE_ASSERT(output.dims != nullptr, "malloc has failed");
    output.dims[0] = 513;
    output.dims[1] = 513;
    output.precision = U8;

    return 0;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    std::cout<< "GetInputInfo Test" << std::endl;
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");
    
    (*info)[0].name = TENSOR_NAME;
    (*info)[0].dimsCount = 4;
    (*info)[0].dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[0].dims) != nullptr, "malloc has failed");
    (*info)[0].dims[0] = 1;
    (*info)[0].dims[1] = 3;
    (*info)[0].dims[2] = 513;
    (*info)[0].dims[3] = 513;
    (*info)[0].precision = FP32;
    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    // // Parameters reading
    int _sourceImageHeight = get_int_parameter("input_h", params, paramsCount, -1);
    int _sourceImageWidth = get_int_parameter("input_w", params, paramsCount, -1);
    NODE_ASSERT(_sourceImageHeight > 0 || _sourceImageHeight == -1, "Source image height - when specified, must be larger than 0");
    NODE_ASSERT(_sourceImageWidth > 0 || _sourceImageHeight == -1, "Source image width - when specified, must be larger than 0");

    int _numClass = get_int_parameter("num_class", params, paramsCount, -1);
    NODE_ASSERT(_numClass > 0, "Number of classes - must be larger than 0");

    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");

    (*info)[0].name = TENSOR_NAME;
    (*info)[0].dimsCount = 2;
    (*info)[0].dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[0].dims) != nullptr, "malloc has failed");
    (*info)[0].dims[0] = 513;
    (*info)[0].dims[1] = 513;
    (*info)[0].precision = U8;

    return 0;
}

int release(void* ptr, void* customNodeLibraryInternalManager) {
    if (ptr != nullptr){
        free(ptr);
    }
    ptr = nullptr;
    return 0;
}
