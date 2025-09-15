//
// Created by rm_autoaim on 2025/9/14.
//

#ifndef RM_AUTO_AIM_ARMOR_HPP
#define RM_AUTO_AIM_ARMOR_HPP

#include <opencv2/core.hpp>

//stl
#include <algorithm>
#include <string>

namespace rm_auto_aim {

    const int RED = 1;
    const int BLUE = 0 ;


    struct Armor {
        Armor() = default;

        cv::Point2f center;
        cv::Mat number_img;
        std::string number;
        float confidence;
        std::string classfication_result;
        //
        float class_scores;
        cv::Rect box;
        cv::Point2f objects_keypoints[4];
        int class_ids;//color
    };
    struct Armors{
        std::vector<float> class_scores;
        std::vector<cv::Rect> boxes;
        std::vector<std::vector<float>> objects_keypoints;
        int class_ids;

    };

    struct dataImg {
        cv::Mat blob;     // 模型输入
        cv::Mat input;    // 原始图像
        float r;          // 缩放比例
        int dw;           // x方向padding
        int dh;           // y方向padding
    };




}


#endif //RM_AUTO_AIM_ARMOR_HPP