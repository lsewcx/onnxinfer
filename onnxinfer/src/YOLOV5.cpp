#include "../include/YOLOV5.hpp"

YOLOv5::YOLOv5(Configuration config)
{
    this->confThreshold = config.confThreshold;
    this->nmsThreshold = config.nmsThreshold;
    this->objThreshold = config.objThreshold;
    this->num_classes = sizeof(this->classes) / sizeof(this->classes[0]); // 类别数量
    this->inpHeight = 640;
    this->inpWidth = 640;

    string model_path = config.modelpath;

    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC); // 设置图优化类型
    ort_session = new Session(env, (const char *)model_path.c_str(), sessionOptions);
    size_t numInputNodes = ort_session->GetInputCount(); // 输入输出节点数量
    size_t numOutputNodes = ort_session->GetOutputCount();
    AllocatorWithDefaultOptions allocator; // 配置输入输出节点内存
    for (int i = 0; i < numInputNodes; i++)
    {
        Ort::AllocatedStringPtr input_name = ort_session->GetInputNameAllocated(i, allocator);
        // input_names.push_back(input_name.get());
        Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_node_dims.push_back(input_dims);
    }

    for (int i = 0; i < numOutputNodes; i++)
    {
        Ort::AllocatedStringPtr output_name = ort_session->GetOutputNameAllocated(i, allocator);
        // output_names.push_back(output_name.get());
        Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_node_dims.push_back(output_dims);
    }
    input_names.push_back("images");
    output_names.push_back("output0");
    this->inpHeight = input_node_dims[0][2];
    this->inpWidth = input_node_dims[0][3];
    this->nout = output_node_dims[0][2];         // 5+classes
    this->num_proposal = output_node_dims[0][1]; // pre_box
}

Mat YOLOv5::resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left)
{
    int srch = srcimg.rows, srcw = srcimg.cols;
    *newh = this->inpHeight;
    *neww = this->inpWidth;
    Mat dstimg;
    if (this->keep_ratio && srch != srcw)
    {
        float hw_scale = (float)srch / srcw;
        if (hw_scale > 1)
        {
            *newh = this->inpHeight;
            *neww = int(this->inpWidth / hw_scale);
            resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
            *left = int((this->inpWidth - *neww) * 0.5);
            copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, BORDER_CONSTANT, 114);
        }
        else
        {
            *newh = (int)this->inpHeight * hw_scale;
            *neww = this->inpWidth;
            resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
            *top = (int)(this->inpHeight - *newh) * 0.5;
            copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 114);
        }
    }
    else
    {
        resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
    }
    return dstimg;
}

void YOLOv5::normalize_(Mat img)
{
    int row = img.rows;
    int col = img.cols;
    this->input_image_.resize(row * col * img.channels()); // vector大小
    for (int c = 0; c < 3; c++)                            // bgr
    {
        for (int i = 0; i < row; i++) // 行
        {
            for (int j = 0; j < col; j++) // 列
            {
                float pix = img.ptr<uchar>(i)[j * 3 + 2 - c]; // Mat里的ptr函数访问任意一行像素的首地址,2-c:表示rgb
                this->input_image_[c * row * col + i * col + j] = pix / 255.0;
            }
        }
    }
}

void YOLOv5::nms(vector<BoxInfo> &input_boxes)
{
    sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b)
         { return a.score > b.score; }); // 降序排列
    vector<float> vArea(input_boxes.size());
    for (int i = 0; i < input_boxes.size(); ++i)
    {
        vArea[i] = (input_boxes[i].x2 - input_boxes[i].x1 + 1) * (input_boxes[i].y2 - input_boxes[i].y1 + 1);
    }
    // 全初始化为false，用来作为记录是否保留相应索引下pre_box的标志vector
    vector<bool> isSuppressed(input_boxes.size(), false);
    for (int i = 0; i < input_boxes.size(); ++i)
    {
        if (isSuppressed[i])
        {
            continue;
        }
        for (int j = i + 1; j < input_boxes.size(); ++j)
        {
            if (isSuppressed[j])
            {
                continue;
            }
            float xx1 = max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = min(input_boxes[i].y2, input_boxes[j].y2);

            float w = max(0.0f, xx2 - xx1 + 1);
            float h = max(0.0f, yy2 - yy1 + 1);
            float inter = w * h; // 交集
            if (input_boxes[i].label == input_boxes[j].label)
            {
                float ovr = inter / (vArea[i] + vArea[j] - inter); // 计算iou
                if (ovr >= this->nmsThreshold)
                {
                    isSuppressed[j] = true;
                }
            }
        }
    }
    // return post_nms;
    int idx_t = 0;
    // remove_if()函数 remove_if(beg, end, op) //移除区间[beg,end)中每一个“令判断式:op(elem)获得true”的元素
    input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo &f)
                                { return isSuppressed[idx_t++]; }),
                      input_boxes.end());
}

void YOLOv5::detect(Mat &frame, bool &draw)
{
    int newh = 0, neww = 0, padh = 0, padw = 0;
    Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw);
    this->normalize_(dstimg);
    // 定义一个输入矩阵，int64_t是下面作为输入参数时的类型
    array<int64_t, 4> input_shape_{1, 3, this->inpHeight, this->inpWidth};
    // 创建输入tensor
    auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

    // 开始推理
    vector<Value> ort_outputs = ort_session->Run(RunOptions{nullptr}, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size()); // 开始推理
    // generate proposals
    vector<BoxInfo> generate_boxes; // BoxInfo自定义的结构体
    float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;
    float *pdata = ort_outputs[0].GetTensorMutableData<float>(); // GetTensorMutableData
    for (int i = 0; i < num_proposal; ++i)                       // 遍历所有的num_pre_boxes
    {
        int index = i * nout;              // prob[b*num_pred_boxes*(classes+5)]
        float obj_conf = pdata[index + 4]; // 置信度分数
        if (obj_conf > this->objThreshold) // 大于阈值
        {
            int class_idx = 0;
            float max_class_socre = 0;
            for (int k = 0; k < this->num_classes; ++k)
            {
                if (pdata[k + index + 5] > max_class_socre)
                {
                    max_class_socre = pdata[k + index + 5];
                    class_idx = k;
                }
            }
            max_class_socre *= obj_conf;               // 最大的类别分数*置信度
            if (max_class_socre > this->confThreshold) // 再次筛选
            {
                // const int class_idx = classIdPoint.x;
                float cx = pdata[index];     // x
                float cy = pdata[index + 1]; // y
                float w = pdata[index + 2];  // w
                float h = pdata[index + 3];  // h

                float xmin = (cx - padw - 0.5 * w) * ratiow;
                float ymin = (cy - padh - 0.5 * h) * ratioh;
                float xmax = (cx - padw + 0.5 * w) * ratiow;
                float ymax = (cy - padh + 0.5 * h) * ratioh;

                generate_boxes.push_back(BoxInfo{xmin, ymin, xmax, ymax, max_class_socre, class_idx});
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    nms(generate_boxes);
    for (size_t i = 0; i < generate_boxes.size(); ++i)
    {
        int xmin = int(generate_boxes[i].x1);
        int ymin = int(generate_boxes[i].y1);
        if (draw)
        {
            rectangle(frame, Point(xmin, ymin), Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), Scalar(0, 0, 255), 2);
            string label = format("%.2f", generate_boxes[i].score);
            label = this->classes[generate_boxes[i].label] + ":" + label;
            putText(frame, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
        }
    }
}