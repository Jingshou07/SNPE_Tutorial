#include <YOLOv8s.h>
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat img = cv::imread("../imgs/bus.jpg");
    cv::Mat img2;
    cv::cvtColor(img, img2, cv::COLOR_BGR2RGB);
    ObjectDetection detect;
    ObjectDetectionConfig cfg;
    cfg.model_path = std::string("../models/yolov8s_200-epochs_640x640_snpe1.61_quantize.dlc");
    cfg.runtime = runtime::DSP;
    cfg.inputLayers = {"images"};
    cfg.outputLayers = {"Split_284", "Mul_326"};
    cfg.outputTensors = {"439", "489"};
    detect.Initialize(cfg);
    std::vector<ObjectData> results;
    detect.Detect(img2, results);
    for (auto i:results) {
        printf("I Got [%d %d %d %d] [%d]:[%f]\n", i.bbox.x, i.bbox.y, i.bbox.width, i.bbox.height, i.label, i.confidence);
        cv::putText(img, std::to_string(i.label)+std::string(" : ")+std::to_string(i.confidence), cv::Point(i.bbox.x, i.bbox.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);
        cv::rectangle(img, cv::Rect(i.bbox.x, i.bbox.y, i.bbox.width, i.bbox.height), cv::Scalar(0, 0, 255), 2);
    }
    cv::imwrite("result.jpg", img);
    printf("I Img saved result.jpg\n");
    return 0;
}