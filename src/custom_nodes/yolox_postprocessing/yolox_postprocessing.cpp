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

struct Object {
    cv::Rect_<float> box;
    float score;
    int class_id;
};

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.box & b.box;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].score;

    while (i <= j)
    {
        while (faceobjects[i].score > p)
            i++;

        while (faceobjects[j].score < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].box.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

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

    float _nmsThresh = get_float_parameter("nms_thresh", params, paramsCount, -1);
    NODE_ASSERT(_nmsThresh > 0 || _nmsThresh <= 1, "NMS Threshold is bitween 0 and 1");

    float _bboxConfThresh = get_float_parameter("bbox_conf_thresh", params, paramsCount, -1);
    NODE_ASSERT(_bboxConfThresh > 0 || _bboxConfThresh <=1, "BBOX Confidence Threshold is bitween 0 and 1");

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
    NODE_ASSERT(imageTensor->dimsCount == 3, "image tensor shape must have 4 dimensions")
    NODE_ASSERT(imageTensor->dims[0] == 1, "image tensor must have batch size equal to 1")

    uint64_t inputNumBoxes = 0;
    uint64_t inputNumAttirib = 0;
    inputNumBoxes = imageTensor->dims[1];
    inputNumAttirib = imageTensor->dims[2];
    
    NODE_ASSERT(inputNumBoxes > 0 && inputNumAttirib > 0, "preds output shape must be positive");
    NODE_ASSERT(inputNumAttirib == (uint64_t)_numClass + 5 , "The number of Attribute must be the sum of the number of class, 1(obj score) and 4(bbox size). ");
    //입력 사이즈에 대한 boxes개수를 계산할 수 있다면 체크하기.

    

    if (debugMode) {
        std::cout << "source image height : " << _sourceImageHeight    << std::endl;
        std::cout << "source image height : " << _sourceImageWidth     << std::endl;
        std::cout << "number of class : "     << _numClass             << std::endl;
        std::cout << "nms threshold "       << _nmsThresh          << std::endl;
        std::cout << "bbox conf threshold " << _bboxConfThresh       << std::endl;
        std::cout << "input shape[0] "      << imageTensor->dims[0] << std::endl;
        std::cout << "input shape[1] "      << inputNumBoxes << std::endl;
        std::cout << "input shape[2] "      << inputNumAttirib << std::endl;
    }
    // // ------------- validation end ---------------

    const float* output_buffer = (float*)imageTensor->data;
    std::vector<Object> objects;

    // net_pred -> output_buffer
    // decode_output (pred, objects, scale, img_w, img_h)
    std::vector<Object> proposals;
    std::vector<int> strides = {8, 16, 32};
    std::vector<GridAndStride> grid_strides;

    // generate_grids_and_stride(INPUT_W, INPUT_H, strides, grid_strides);
    for (auto stride : strides)
    {
        int num_grid_w = _sourceImageWidth / stride;
        int num_grid_h = _sourceImageHeight / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                grid_strides.push_back((GridAndStride){g0, g1, stride});
                // if(debugMode){
                //     std::cout << "Grid And Stride : " << "g0 : " << g0 <<  ", g1 : " << g1 << ", stride : " << stride << std::endl;
                // }
            }
        }
    }

    // generate_yolox_proposals(grid_stirdes, pred, BBOX_CONF_THRESH, proposals)
    const int num_anchors = grid_strides.size();
    // std::cout << "NUM_ANCHORS : " << num_anchors << std::endl;

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;
        // std::cout << "AnchorIdx(" << anchor_idx << ")" << grid0 << ", " << grid1 << ", " << stride;
        // std::endl;

        const int basic_pos = anchor_idx * (_numClass + 5);

        // yolox/models/yolo_head.py decode logic
        //  outputs[..., :2] = (outputs[..., :2] + grids) * strides
        //  outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        float x_center = (output_buffer[basic_pos + 0] + grid0) * stride;
        float y_center = (output_buffer[basic_pos + 1] + grid1) * stride;
        float w = exp(output_buffer[basic_pos + 2]) * stride;
        float h = exp(output_buffer[basic_pos + 3]) * stride;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;

        float box_objectness = output_buffer[basic_pos + 4];
        for (int class_idx = 0; class_idx < _numClass; class_idx++)
        {
            float box_cls_score = output_buffer[basic_pos + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > _bboxConfThresh)
            {
                
                Object obj;
                obj.box.x = x0;
                obj.box.y = y0;
                obj.box.width = w;
                obj.box.height = h;
                obj.class_id = class_idx;
                obj.score = box_prob;

                proposals.push_back(obj);

                // std::cout << "Object : " << obj.class_id << " , " << obj.score << std::endl; 
            }

        } // class loop

    } // point anchor loop

    std::cout << "NUM OBJECTS : " << proposals.size() << std::endl;

    //qsort_descent_inplace(proposals)
    qsort_descent_inplace(proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, _nmsThresh);
    int count = picked.size();
    objects.resize(count);

    std::cout << "NMS RESULT OBJECTS : " << count << std::endl;

    float scale = 1.0; // scale -> min( src_width / ori_width, src_height / ori_height )

    int data_depth = 6;
    uint64_t byteSize = sizeof(float) * count * data_depth; // 6 = id, score, x, y, w, h
    float* buffer = (float*)malloc(byteSize);
    NODE_ASSERT(buffer != nullptr, "malloc has failed.");

    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].box.x) / scale;
        float y0 = (objects[i].box.y) / scale;
        float x1 = (objects[i].box.x + objects[i].box.width) / scale;
        float y1 = (objects[i].box.y + objects[i].box.height) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(_sourceImageWidth - 1)), 0.f); // _source size -> original size
        y0 = std::max(std::min(y0, (float)(_sourceImageHeight - 1)), 0.f); // _source size -> original size
        x1 = std::max(std::min(x1, (float)(_sourceImageWidth - 1)), 0.f); // _source size -> original size
        y1 = std::max(std::min(y1, (float)(_sourceImageHeight - 1)), 0.f); // _source size -> original size

        objects[i].box.x = x0;
        objects[i].box.y = y0;
        objects[i].box.width = x1 - x0;
        objects[i].box.height = y1 - y0;

        std::cout << "ID(" << objects[i].class_id << ") score(" << objects[i].score << ") BBOX(" << objects[i].box.x<< ", "<< objects[i].box.y<< ", "<< objects[i].box.width << ", "<< objects[i].box.height<< ")" << std::endl;

        int pos = i * data_depth;
        buffer[pos + 0] = objects[i].class_id;
        buffer[pos + 1] = objects[i].score * 100;
        buffer[pos + 2] = objects[i].box.x;
        buffer[pos + 3] = objects[i].box.y;
        buffer[pos + 4] = objects[i].box.width;
        buffer[pos + 5] = objects[i].box.height;
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
    output.dimsCount = 3;
    output.dims = (uint64_t*)malloc(output.dimsCount * sizeof(uint64_t));
    NODE_ASSERT(output.dims != nullptr, "malloc has failed");
    output.dims[0] = 1;
    output.dims[1] = count;
    output.dims[2] = data_depth;
    output.precision = FP32;
    return 0;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");
    
    (*info)[0].name = TENSOR_NAME;
    (*info)[0].dimsCount = 3;
    (*info)[0].dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[0].dims) != nullptr, "malloc has failed");
    (*info)[0].dims[0] = 1;
    (*info)[0].dims[1] = 3549; // set as input image shape, stride = {8, 16, 32}, sum(width / stride * height / stride)
    (*info)[0].dims[2] = 85; // set as the number of class, 1(obj score) + 4(bbox coord) + num_class
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

    float _nmsThresh = get_float_parameter("nms_thresh", params, paramsCount, -1);
    NODE_ASSERT(_nmsThresh > 0 || _nmsThresh <= 1, "NMS Threshold is bitween 0 and 1");

    float _bboxConfThresh = get_float_parameter("bbox_conf_thresh", params, paramsCount, -1);
    NODE_ASSERT(_bboxConfThresh > 0 || _bboxConfThresh <=1, "BBOX Confidence Threshold is bitween 0 and 1");

    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");

    (*info)[0].name = TENSOR_NAME;
    (*info)[0].dimsCount = 3;
    (*info)[0].dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[0].dims) != nullptr, "malloc has failed");
    (*info)[0].dims[0] = 1;
    (*info)[0].dims[1] = -1;
    (*info)[0].dims[2] = 6;

    (*info)[0].precision = FP32;

    return 0;
}

int release(void* ptr, void* customNodeLibraryInternalManager) {
    if (ptr != nullptr){
        free(ptr);
    }
    ptr = nullptr;
    return 0;
}
