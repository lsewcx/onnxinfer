
#include "../include/YOLOV5.hpp"
#include "Motion.hpp"
// 自定义配置结构

// int endsWith(string s, string sub) {
// 	return s.rfind(sub) == (s.length() - sub.length()) ? 1 : 0;
// }

// const float anchors_640[3][6] = { {10.0,  13.0, 16.0,  30.0,  33.0,  23.0},
// 								 {30.0,  61.0, 62.0,  45.0,  59.0,  119.0},
// 								 {116.0, 90.0, 156.0, 198.0, 373.0, 326.0} };

// const float anchors_1280[4][6] = { {19, 27, 44, 40, 38, 94},{96, 68, 86, 152, 180, 137},{140, 301, 303, 264, 238, 542},
// 					   {436, 615, 739, 380, 925, 792} };

int main(int argc, char *argv[])
{
    Motion motion;
    clock_t startTime, endTime; // 计算时间
    Configuration yolo_nets = {motion.params.confidencethreshold, motion.params.nmsthreshold, motion.params.objthreshold, motion.params.modelpath};
    YOLOv5 yolo_model(yolo_nets);
    Mat srcimg = imread(motion.params.imgpath);
    double timeStart = (double)getTickCount();
    startTime = clock(); // 计时开始
    yolo_model.detect(srcimg, motion.params.draw);
    endTime = clock();                                                               // 计时结束
    double nTime = ((double)getTickCount() - timeStart) / getTickFrequency() * 1000; // 转换为毫秒
    cout << "clock_running time is:" << (double)(endTime - startTime) / CLOCKS_PER_SEC * 1000 << "ms" << endl;
    cout << "The run time is:" << (double)clock() / CLOCKS_PER_SEC * 1000 << "ms" << endl;
    cout << "getTickCount_running time :" << nTime << "ms\n"
         << endl;
    imshow("bus", srcimg);
    // imwrite("bus.jpg", srcimg);
    waitKey(0);
    return 0;
}