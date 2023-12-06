#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <tuple>
#include <list>
#include <ctime>

#include <opencv2/opencv.hpp> 

#include <AlgYolov8s-3rd.h>
#include <YOLOv8s.h>

#include "nlohmann/json.hpp"
using json = nlohmann::json;

typedef struct _AlgConfig        {
    runtime device_         { runtime::DSP };
    float   conf_thresh_    {         0.77 };
    float   nms_thresh_     {          0.5 };
    int32_t min_border_     {           10 };
    int32_t timing_seconds_ {            1 };
} AlgConfig;

typedef struct _AlgCore {
    AlgConfig                        cfg_      ;
    std::shared_ptr<ObjectDetection> detector_ ;
    time_t last_timestamp_           {        };
} AlgCore;

static bool check_event_time(const int32_t& timing_seconds_, time_t& last_timestamp_) {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    if ((tv.tv_sec - last_timestamp_) >= timing_seconds_) {
        last_timestamp_ = tv.tv_sec;
        return true;
    }
    return false;
}


static runtime string_to_device (std::string& device) {
    std::transform (device.begin (), device.end (), device.begin (),
        [] (unsigned char ch) { return tolower (ch); });
    if (0 == device.compare ("cpu")) {
        return runtime::CPU;
    } else if (0 == device.compare ("gpu")) {
        return runtime::GPU;
    } else if (0 == device.compare ("dsp")) {
        return runtime::DSP;
    } else { 
        return runtime::CPU;
    }
}

static bool parse_args (AlgConfig& config, const std::string& data) {
    json cfg = nlohmann::json::parse(data);
    if (cfg.contains("confidence-thresh"))
        config.conf_thresh_    = cfg["confidence-thresh"].get<float>();
    if (cfg.contains("nms-thresh"))
        config.nms_thresh_     = cfg["nms-thresh"].get<float>();
    if (cfg.contains("min-box-border-size"))
        config.min_border_     = cfg["min-box-border-size"].get<int>();
    if (cfg.contains("timing-seconds"))
        config.timing_seconds_ = cfg["timing-seconds"].get<int>();
}

static JsonObject* results_to_json_object (const std::vector<ObjectData>& results) {
    JsonObject* result = json_object_new ();
    if (!result) {
        TS_ERR_MSG_V ("Failed to new an object with type JsonObject*");
        return nullptr;
    }
    json_object_set_string_member (result, "alg-name", "yolov8s-3rd");
    JsonArray* jarray = json_array_new();
    if (!jarray) {
        TS_ERR_MSG_V ("Failed to new an object with type JsonArray*");
        json_object_unref (result);
        return nullptr;
    }
    for (size_t i = 0; i < results.size (); i++) {
        JsonObject* jobject = json_object_new ();
        if (!jobject) {
            TS_ERR_MSG_V ("Failed to new an object with type JsonObject*");
            json_object_unref (result);
            json_array_unref  (jarray);
            return nullptr;
        }
        json_object_set_string_member (jobject, "label",
            std::to_string (results[i].label      ).c_str ());
        json_object_set_string_member (jobject, "x",
            std::to_string (results[i].bbox.x     ).c_str ());
        json_object_set_string_member (jobject, "y",
            std::to_string (results[i].bbox.y     ).c_str ());
        json_object_set_string_member (jobject, "width",
            std::to_string (results[i].bbox.width ).c_str ());
        json_object_set_string_member (jobject, "height",
            std::to_string (results[i].bbox.height).c_str ());
        json_object_set_string_member (jobject, "confidence",
            std::to_string (results[i].confidence ).c_str ());
        json_array_add_object_element (jarray, jobject);
    }
    json_object_set_array_member  (result, "alg-result", jarray);
    return result;
}

static void results_to_osd_object (const std::vector<ObjectData>& results, std::vector<TsOsdObject>& osds) {
    for (auto&& item : results) {
        std::string label = std::to_string(item.label);
        osds.push_back (TsOsdObject (item.bbox.x, item.bbox.y, item.bbox.width, item.bbox.height, 
            255, 0, 0, 0, label, TsObjectType::OBJECT));
    }
}

void* algInit (const std::string& args) {
    TS_INFO_MSG_V ("algInit of ALG named yolov8s-3rd called");
    AlgCore* a = new AlgCore ();
    if (!a) {
        TS_ERR_MSG_V ("Failed to new an object with type AlgCore");
        return nullptr;
    }
    if (0 != args.compare("")) parse_args(a->cfg_, args);

    if (!(a->detector_ = std::make_shared<ObjectDetection> ())) {
        TS_ERR_MSG_V ("Failed to new an object typed ObjectDetection");
        return nullptr;
    }
    ObjectDetectionConfig cfg;
    cfg.model_path = std::string("/opt/thundersoft/algs/model/yolov8s_200-epochs_640x640_snpe1.61_quantize.dlc");
    cfg.runtime = runtime::DSP;
    cfg.inputLayers = {"images"};
    cfg.outputLayers = {"Split_284", "Mul_326"};
    cfg.outputTensors = {"439", "489"};
    a->detector_->Initialize(cfg);
    a->detector_->SetMinBoxBorder(a->cfg_.min_border_);
    a->detector_->SetScoreThresh(a->cfg_.conf_thresh_, a->cfg_.nms_thresh_);
    return (void*) a;
}

bool algStart (void* alg) {
    return true;
}


std::shared_ptr<TsJsonObject> algProc (void* alg, const std::shared_ptr<TsGstSample>& data) {
    AlgCore* a = (AlgCore*) alg;

    GstSample* sample = data->GetSample ();
    GstCaps* caps = gst_sample_get_caps (sample);
    GstStructure* structure = gst_caps_get_structure (caps, 0);
    int width;  gst_structure_get_int (structure, "width",  &width );
    int height; gst_structure_get_int (structure, "height", &height);
    std::string format ((char*) gst_structure_get_string (
        structure, "format"));
    if (0 != format.compare ("RGB") && 0 != format.compare ("BGR")) {
        TS_ERR_MSG_V ("Invalid format(%s!=RGB|BGR)", format.c_str());
        return nullptr;
    }
    GstMapInfo map;
    GstBuffer* buf = gst_sample_get_buffer(sample);
    gst_buffer_map(buf, &map, GST_MAP_READ);
    cv::Mat img (height, width, CV_8UC3, map.data);
    gst_buffer_unmap(buf, &map);

    std::vector<ObjectData> results;
    a->detector_->Detect(img, results);

    std::shared_ptr<TsJsonObject> jo = std::make_shared<TsJsonObject> (results_to_json_object (results));
    results_to_osd_object (results, jo->GetOsdObject ());
    if (results.size() > 0 && check_event_time(a->cfg_.timing_seconds_, a->last_timestamp_)) {
        jo->SetLevel (TsJsonObject::Level::WARNING);
        jo->SetSnapPicture (true);
    } 
    return jo;
}


bool algCtrl (void* alg, const std::string& cmd) {
    return false;
}

void algStop (void* alg) {

}

void algFina(void* alg) {
    AlgCore* a = (AlgCore*) alg;
    a->detector_->DeInitialize ();
    delete a;
}

bool algSetCb (void* alg, TsPutResult cb, void* args) {
    AlgCore* a = (AlgCore*) alg;
    return false;
}

