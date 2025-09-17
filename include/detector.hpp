//
// Created by rm_autoaim on 2025/9/14.
//

#ifndef RM_AUTO_AIM_DETECTOR_HPP
#define RM_AUTO_AIM_DETECTOR_HPP
//opencv
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
//std
#include <cmath>
#include <string>
#include <vector>

#include "armor.hpp"
#include "number_classifier.hpp"

namespace rm_auto_aim {
    class Detector {
    public:
        Detector(const std::string &model_path, const float score_threshold, const float nms_threshold,
                 const int detector_color)
            : model_path(model_path), score_threshold(score_threshold), nms_threshold(nms_threshold),
              detector_color(detector_color) {
        }

        std::vector<Armor> detect(const cv::Mat &input);

        dataImg preprocessImageDate(const cv::Mat &img, cv::Size new_shape = cv::Size(416, 416),
                                    cv::Scalar color = cv::Scalar(114, 114, 114)); //图片预处理
        std::vector<Armor> startInferAndNMS(dataImg data); //开始推理并返回结果
        cv::Mat getAllNumbersImage();

        void drawResults(const cv::Mat &img);

        std::string model_path;
        float score_threshold;
        float nms_threshold;


        int detector_color;

        std::unique_ptr<NumberClassifier> classifier;

        //debug
        cv::Mat binary_img;
        // auto_aim_interfaces::msg::DebugLights debug_lights;
        // auto_aim_interfaces::msg::DebugArmors debug_armors;

    private:
        std::vector<Armor> armors_;
    };
}


#endif //RM_AUTO_AIM_DETECTOR_HPP
