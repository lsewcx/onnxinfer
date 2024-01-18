#include <fstream>
#include <sstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cpu_provider_factory.h> // 提供cuda加速
#include <onnxruntime_cxx_api.h>  // C或c++的api
#include<iostream>
using namespace std;
using namespace cv;
using namespace Ort;


