#include <math.h>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "YOLOv8s.h"

static float sigmoid(float x) { 
    return 1.0 / (1.0 + expf(-x));
}

static float unsigmoid(float y) { 
    return -1.0 * logf((1.0 / y) - 1.0);
}

ObjectDetection::ObjectDetection() : m_task(nullptr) {

}

ObjectDetection::~ObjectDetection() {
    DeInitialize();
}

bool ObjectDetection::Initialize(const ObjectDetectionConfig& config) {
    m_task = std::move(std::unique_ptr<snpetask::SNPETask>(new snpetask::SNPETask()));
    m_inputLayers = config.inputLayers;
    m_outputLayers = config.outputLayers;
    m_outputTensors = config.outputTensors;
    m_labels = config.labels;
    m_grids = config.grids;
    m_task->setOutputLayers(m_outputLayers);

    if (!m_task->init(config.model_path, config.runtime)) {
        printf("ERROR: Can't init snpetask instance.\n");
        return false;
    }

    m_output = new float[m_grids * m_labels];
    m_isInit = true;
    return true;
}

bool ObjectDetection::DeInitialize() {
    if (m_task) {
        m_task->deInit();
        m_task.reset(nullptr);
    }
    if (m_output) {
        delete[] m_output;
        m_output = nullptr;
    }
    m_isInit = false;
    return true;
}

bool ObjectDetection::PreProcess(const cv::Mat& image) {
    auto inputShape = m_task->getInputShape(m_inputLayers[0]);
    size_t batch = inputShape[0];
    size_t inputHeight = inputShape[1];
    size_t inputWidth = inputShape[2];
    size_t channel = inputShape[3];

    if (m_task->getInputTensor(m_inputLayers[0]) == nullptr) {
        printf("ERROR: Empty input tensor\n");
        return false;
    }

    cv::Mat input(inputHeight, inputWidth, CV_32FC3, m_task->getInputTensor(m_inputLayers[0]));
    if (image.empty()) {
        printf("ERROR: Invalid image!\n");
        return false;
    }
    int imgWidth = image.cols;
    int imgHeight = image.rows;

    m_scale = std::min(inputHeight /(float)imgHeight, inputWidth / (float)imgWidth);
    int scaledWidth = imgWidth * m_scale;
    int scaledHeight = imgHeight * m_scale;
    m_xOffset = (inputWidth - scaledWidth) / 2;
    m_yOffset = (inputHeight - scaledHeight) / 2;

    cv::Mat inputMat(inputHeight, inputWidth, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat roiMat(inputMat, cv::Rect(m_xOffset, m_yOffset, scaledWidth, scaledHeight));
    cv::resize(image, roiMat, cv::Size(scaledWidth, scaledHeight), cv::INTER_LINEAR);
    inputMat.convertTo(input, CV_32FC3);
    input /= 255.0f;
}

bool ObjectDetection::Detect(const cv::Mat& image, std::vector<ObjectData>& results) {
    PreProcess(image);
    int64_t start = GetTimeStamp_ms();
    if (!m_task->execute()) {
        printf("ERROR: SNPETask execute failed.\n");
        return false;
    }
    PostProcess(results, GetTimeStamp_ms() - start);
    return true;
}

bool ObjectDetection::PostProcess(std::vector<ObjectData> &results, int64_t time) {
    auto outputShape = m_task->getOutputShape(m_outputTensors[0]);
    const float *predOutput = m_task->getOutputTensor(m_outputTensors[0]);
    const float *output = m_task->getOutputTensor(m_outputTensors[1]);
    int B_ = outputShape[0];
    int H_ = outputShape[1];
    int W_ = outputShape[2];

    std::vector<int> boxIndexs;
    std::vector<float> boxConfidences;
    std::vector<ObjectData> winList;

    for (int i=0; i<W_; i++) {
        float maxScore = -110.0f;
        int   maxIndex = -1;
        for (int j=0; j<H_; j++) {
            if (*(predOutput+j*W_+i) > maxScore) {
                maxScore = *(predOutput+j*W_+i);
                maxIndex = j;
            }
        }
        if (sigmoid(maxScore) > m_confThresh) {
            float x = *(output+0*W_+i);
            float y = *(output+1*W_+i);
            float w = *(output+2*W_+i);
            float h = *(output+3*W_+i);
            x = x-0.5*w;
            y = y-0.5*h;
            x -= m_xOffset;
            y -= m_yOffset;
            w /= m_scale;
            h /= m_scale;
            x /= m_scale;
            y /= m_scale;
            
            ObjectData rect;
            rect.bbox.x = x;
            rect.bbox.y = y;
            rect.bbox.width = w;
            rect.bbox.height = h;
            rect.label = maxIndex;
            rect.confidence = sigmoid(maxScore);
            winList.push_back(rect);
        }
    }

    winList = nms(winList, m_nmsThresh);
    for (size_t i = 0; i < winList.size(); i++) {
        if (winList[i].bbox.width >= m_minBoxBorder || winList[i].bbox.height >= m_minBoxBorder) {
            results.push_back(winList[i]);
        }
    }
}
