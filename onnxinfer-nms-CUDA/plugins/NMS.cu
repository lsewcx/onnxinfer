#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include "../include/cuda_utils.h"
#include "NMS.h"

__global__ void nms_kernel(BoxInfo *input_boxes, bool *isSuppressed, int n, float nmsThreshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float area_i = (input_boxes[i].x2 - input_boxes[i].x1 + 1) * (input_boxes[i].y2 - input_boxes[i].y1 + 1);
        for (int j = i + 1; j < n; ++j)
        {
            float xx1 = max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = min(input_boxes[i].y2, input_boxes[j].y2);

            float w = max(0.0f, xx2 - xx1 + 1);
            float h = max(0.0f, yy2 - yy1 + 1);
            float inter = w * h; // 交集
            if (input_boxes[i].label == input_boxes[j].label)
            {
                float area_j = (input_boxes[j].x2 - input_boxes[j].x1 + 1) * (input_boxes[j].y2 - input_boxes[j].y1 + 1);
                float ovr = inter / (area_i + area_j - inter); // 计算iou
                if (ovr >= nmsThreshold)
                {
                    isSuppressed[j] = true;
                }
            }
        }
    }
}

void nmscuda(std::vector<BoxInfo> &input_boxes_host)
{
    thrust::device_vector<BoxInfo> input_boxes = input_boxes_host;
    thrust::device_vector<bool> isSuppressed(input_boxes.size());

    thrust::fill(isSuppressed.begin(), isSuppressed.end(), false);

    int blockSize = 256;
    int numBlocks = (input_boxes.size() + blockSize - 1) / blockSize;
    float nmsThreshold = 0.3;
    nms_kernel<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(input_boxes.data()), thrust::raw_pointer_cast(isSuppressed.data()), input_boxes.size(), nmsThreshold);

    // 将结果从设备内存复制回主机内存
    thrust::copy(input_boxes.begin(), input_boxes.end(), input_boxes_host.begin());
    std::vector<bool> isSuppressed_host(isSuppressed.size());
    thrust::copy(isSuppressed.begin(), isSuppressed.end(), isSuppressed_host.begin());

    // 在主机内存中删除被抑制的框
    int idx_t = 0;
    input_boxes_host.erase(remove_if(input_boxes_host.begin(), input_boxes_host.end(), [&idx_t, &isSuppressed_host](const BoxInfo &f)
                                     { return isSuppressed_host[idx_t++]; }),
                           input_boxes_host.end());
}