//
// Created by rm_autoaim on 2025/9/14.
//

#include "../include/detector.hpp"
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

// STD
#include <algorithm>
#include <cmath>
#include <vector>
#include <mutex>
#include <oneapi/tbb/detail/_config.h>

#include <openvino/openvino.hpp>


namespace rm_auto_aim {
    static void sort_keypoints(cv::Point2f keypoints[4]) {
        // Sort points based on their y-coordinates (ascending)
        std::sort(keypoints, keypoints + 4, [](const cv::Point &a, const cv::Point &b) {
            return a.y < b.y;
        });

        // Top points will be the first two, bottom points will be the last two
        cv::Point top_points[2] = {keypoints[0], keypoints[1]};
        cv::Point bottom_points[2] = {keypoints[2], keypoints[3]};

        // Sort the top points by their x-coordinates to distinguish left and right
        std::sort(top_points, top_points + 2, [](const cv::Point &a, const cv::Point &b) {
            return a.x < b.x;
        });

        // Sort the bottom points by their x-coordinates to distinguish left and right
        std::sort(bottom_points, bottom_points + 2, [](const cv::Point &a, const cv::Point &b) {
            return a.x < b.x;
        });

        // Assign sorted points back to the keypoints array
        keypoints[0] = top_points[0]; // top-left
        keypoints[1] = bottom_points[0]; // bottom-left
        keypoints[2] = bottom_points[1]; // bottom-right
        keypoints[3] = top_points[1]; // top-right
    }


    dataImg Detector::preprocessImageDate(const cv::Mat &img, cv::Size new_shape, cv::Scalar color) {
        cv::Size shape = img.size();

        // 缩放比例
        float r = std::min((float)new_shape.height / shape.height,
                           (float)new_shape.width / shape.width);

        // 缩放后的大小
        cv::Size new_unpad(int(round(shape.width * r)),
                           int(round(shape.height * r)));

        // padding
        int dw = new_shape.width - new_unpad.width;
        int dh = new_shape.height - new_unpad.height;

        // resize
        cv::Mat resized_img;
        cv::resize(img, resized_img, new_unpad);

        // padding （注意：分左右上下）
        int top = dh / 2;
        int bottom = dh - top;
        int left = dw / 2;
        int right = dw - left;

        cv::Mat bordered_img;
        cv::copyMakeBorder(resized_img, bordered_img,
                           top, bottom, left, right,
                           cv::BORDER_CONSTANT, color);

        // blob
        cv::Mat blob = cv::dnn::blobFromImage(
            bordered_img, 1.0 / 255.0,
            new_shape, cv::Scalar(), true, false);

        // 封装输出
        dataImg data;
        data.blob = blob;
        data.input = img;
        data.r = r;
        data.dw = left;
        data.dh = top;

        std::cout << "[Preprocess] original=" << shape
                  << " resized=" << new_unpad
                  << " padded=" << new_shape
                  << " r=" << r
                  << " dw=" << data.dw
                  << " dh=" << data.dh << std::endl;

        return data;
    }



    std::vector<Armor> Detector::detect(const cv::Mat &input) {

        auto blob = preprocessImageDate(input);

        armors_ = startInferAndNMS(blob);

        if (!armors_.empty()) {
            std::cout<<"find armor"<<std::endl;
            classifier->extractNumbers(input, armors_);
            classifier->classify(armors_);

            drawResults(input);
        }
        return armors_;
    }

    std::vector<Armor> Detector::startInferAndNMS(dataImg img_data) {
    static std::once_flag init_flag;
    static ov::Core core;
    static ov::CompiledModel compiled_model;
    static ov::InferRequest infer_request;
    static ov::Output<const ov::Node> input_port;

    // 模型初始化只执行一次
    std::call_once(init_flag, [&]() {
        std::cout << "initialize OpenVINO model..." << std::endl;
        compiled_model = core.compile_model(Detector::model_path, "CPU");
        infer_request = compiled_model.create_infer_request();
        input_port = compiled_model.input();
    });

    // 构造输入张量
    ov::Tensor input_tensor(input_port.get_element_type(),
                            input_port.get_shape(),
                            img_data.blob.ptr(0));
    infer_request.set_input_tensor(input_tensor);

    // -------- 开始推理 --------
    infer_request.infer();

    // -------- 获取输出 --------
    auto output = infer_request.get_output_tensor(0);
    auto output_shape = output.get_shape();
    std::cout << "[Infer] Output tensor shape:";
    for (auto s : output_shape) std::cout << " " << s;
    std::cout << std::endl;

    float *data = output.data<float>();
    cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, data);
    transpose(output_buffer, output_buffer); // [num, 14]

    std::vector<Armor> qualifiedArmors;

    // 遍历类别（假设 class=2 -> id=4,5）
    for (int cls = 4; cls < 6; ++cls) {
        Armors SingleData;

        for (int i = 0; i < output_buffer.rows; i++) {
            float class_score = output_buffer.at<float>(i, cls);

            // 找最大类别分数
            float max_class_score = 0.0;
            for (int j = 4; j < 6; j++) {
                max_class_score = std::max(max_class_score, output_buffer.at<float>(i, j));
            }
            if (class_score != max_class_score) continue;
            if (class_score < Detector::score_threshold) continue;

            SingleData.class_scores.push_back(class_score);

            // 预测框
            float cx = output_buffer.at<float>(i, 0);
            float cy = output_buffer.at<float>(i, 1);
            float w  = output_buffer.at<float>(i, 2);
            float h  = output_buffer.at<float>(i, 3);

            // 从 letterbox 图还原回原图
            float x0 = (cx - 0.5f * w - img_data.dw) / img_data.r;
            float y0 = (cy - 0.5f * h - img_data.dh) / img_data.r;
            float x1 = (cx + 0.5f * w - img_data.dw) / img_data.r;
            float y1 = (cy + 0.5f * h - img_data.dh) / img_data.r;

            int left   = std::clamp((int)x0, 0, img_data.input.cols - 1);
            int top    = std::clamp((int)y0, 0, img_data.input.rows - 1);
            int right  = std::clamp((int)x1, 0, img_data.input.cols - 1);
            int bottom = std::clamp((int)y1, 0, img_data.input.rows - 1);

            SingleData.boxes.emplace_back(left, top, right - left, bottom - top);

            // 关键点
            std::vector<float> keypoints;
            cv::Mat kpts = output_buffer.row(i).colRange(6, 14);
            for (int j = 0; j < 4; j++) {
                float x = (kpts.at<float>(0, j*2 + 0) - img_data.dw) / img_data.r;
                float y = (kpts.at<float>(0, j*2 + 1) - img_data.dh) / img_data.r;
                x = std::clamp(x, 0.0f, (float)img_data.input.cols - 1);
                y = std::clamp(y, 0.0f, (float)img_data.input.rows - 1);
                keypoints.push_back(x);
                keypoints.push_back(y);
            }
            SingleData.objects_keypoints.push_back(keypoints);
        }

        SingleData.class_ids = cls - 4;

        // NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(SingleData.boxes, SingleData.class_scores,
                          Detector::score_threshold, Detector::nms_threshold, indices);

        for (auto i : indices) {
            Armor armor;
            armor.box = SingleData.boxes[i];
            armor.class_scores = SingleData.class_scores[i];
            armor.class_ids = SingleData.class_ids;

            for (int j = 0; j < 4; j++) {
                armor.objects_keypoints[j] = cv::Point(
                    (int)SingleData.objects_keypoints[i][j*2 + 0],
                    (int)SingleData.objects_keypoints[i][j*2 + 1]
                );
            }
            sort_keypoints(armor.objects_keypoints);

            if (armor.class_ids != detector_color) {
                continue;
            }

            qualifiedArmors.push_back(armor);
        }

        std::cout << "[Post] Class " << cls
                  << ": before NMS = " << SingleData.boxes.size()
                  << ", after NMS = " << indices.size() << std::endl;
    }

    std::cout << "[Post] Total qualified armors = "
              << qualifiedArmors.size() << std::endl;

    return qualifiedArmors;
}

    cv::Mat Detector::getAllNumbersImage() {
        if (armors_.empty()) {
            return cv::Mat(cv::Size(20, 28),CV_8UC1);
        } else {
            std::vector<cv::Mat> number_imgs;
            number_imgs.reserve(armors_.size());
            for (auto armor: armors_) {
                number_imgs.emplace_back(armor.number_img);
            }
            cv::Mat all_num_img;
            cv::vconcat(number_imgs, all_num_img);
            return all_num_img;
        }
    }


    void Detector::drawResults(const cv::Mat &src) {
        for (const auto &armor : armors_) {
            // 绘制检测框
            cv::rectangle(src, armor.box, cv::Scalar(0, 255, 0), 2);

            // 绘制四个关键点
            for (const auto &pt : armor.objects_keypoints) {
                cv::circle(src, pt, 3, cv::Scalar(0, 0, 255), -1);
            }

            // 颜色文字
            std::string color_text;
            switch (armor.class_ids) {
                case RED: color_text = "RED"; break;
                case BLUE: color_text = "BLUE"; break;
                default: color_text = "UNK"; break;
            }

            // 分类结果（数字 + 置信度）
            std::string text = color_text + " " + armor.classfication_result;

            // 背景框
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::Point text_origin(armor.box.x, std::max(0, armor.box.y - 5));

            cv::rectangle(src, text_origin + cv::Point(0, baseline),
                          text_origin + cv::Point(text_size.width, -text_size.height),
                          cv::Scalar(0, 255, 0), cv::FILLED);


            cv::Scalar font_color = (armor.class_ids == RED) ? cv::Scalar(0,0,255) :
                                    (armor.class_ids == BLUE) ? cv::Scalar(255,0,0) :
                                    cv::Scalar(0,0,0);

            cv::putText(src, text, text_origin, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        font_color, 1, cv::LINE_AA);
        }
    }




}
