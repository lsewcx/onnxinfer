
#ifndef COMMON_H
#define COMMON_H

#include <fstream>
#include <sstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h> // C或c++的api
#include <iostream>
using namespace std;
using namespace cv;
using namespace Ort;

struct Configuration
{
public:
    float confThreshold; // Confidence threshold
    float nmsThreshold;  // Non-maximum suppression threshold
    float objThreshold;  // Object Confidence threshold
    string modelpath;
};

// 定义BoxInfo结构类型
typedef struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

#endif // COMMON_H