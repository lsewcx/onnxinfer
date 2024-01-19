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

void nmscuda(vector<BoxInfo> &input_boxes)
{
    // 将输入数据从主机内存复制到设备内存
    BoxInfo *d_input_boxes;
    cudaMalloc(&d_input_boxes, input_boxes.size() * sizeof(BoxInfo));
    cudaMemcpy(d_input_boxes, input_boxes.data(), input_boxes.size() * sizeof(BoxInfo), cudaMemcpyHostToDevice);

    // 初始化isSuppressed数组
    bool *d_isSuppressed;
    cudaMalloc(&d_isSuppressed, input_boxes.size() * sizeof(bool));
    cudaMemset(d_isSuppressed, 0, input_boxes.size() * sizeof(bool));

    // 调用CUDA内核函数
    int blockSize = 256;
    int numBlocks = (input_boxes.size() + blockSize - 1) / blockSize;
    int nmsThreshold = 0.3;
    nms_kernel<<<numBlocks, blockSize>>>(d_input_boxes, d_isSuppressed, input_boxes.size(), nmsThreshold);

    // 将结果从设备内存复制回主机内存
    cudaMemcpy(input_boxes.data(), d_input_boxes, input_boxes.size() * sizeof(BoxInfo), cudaMemcpyDeviceToHost);
    bool *isSuppressed = new bool[input_boxes.size()];
    cudaMemcpy(isSuppressed, d_isSuppressed, input_boxes.size() * sizeof(bool), cudaMemcpyDeviceToHost);

    // 清理设备内存
    cudaFree(d_input_boxes);
    cudaFree(d_isSuppressed);

    // 在主机内存中删除被抑制的框
    int idx_t = 0;
    input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, isSuppressed](const BoxInfo &f)
                                { return isSuppressed[idx_t++]; }),
                      input_boxes.end());
    delete[] isSuppressed;
}