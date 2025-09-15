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

    enum class ArmorType {SMALL, LARGE, INVALID};  //zeng qiang zuo yong yu
    const std::string ARMOR_TYPE_STR[3] = {"small", "large", "invalid"};

    struct Light : public cv::Rect {
        Light() = default;
        explicit Light(cv::Rect box, cv::Point2f top, cv::Point2f bottom,int area, float tilt_angle
        ):cv::Rect(box),top(top),bottom(bottom),tilt_angle(tilt_angle)
        {
            length = cv::norm(top - bottom);
            width = area / length;
            center = (top + bottom) / 2;
        }
        int color;
        cv::Point2f top, bottom;
        cv::Point2f center;
        double length;
        double width;
        float tilt_angle;
    };
    struct Armor {
        Armor() = default;
        Armor(const Light & l1, const Light &l2) {
            left_light = (l1.center.x < l2.center.x) ? l1 : l2;
            right_light = (l1.center.x < l2.center.x) ? l2 : l1;
        }

        Light left_light, right_light;
        cv::Point2f center;
        ArmorType type;

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