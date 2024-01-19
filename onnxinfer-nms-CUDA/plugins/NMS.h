#ifndef NMS_H
#define NMS_H

#include <vector>
#include "../include/common.hpp"
// 定义BoxInfo结构类型

// 主机函数的声明
void nmscuda(std::vector<BoxInfo> &input_boxes);

#endif // NMS_H