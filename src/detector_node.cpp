
#include "detector_node.hpp"

namespace rm_auto_aim {
    ArmorDetectorNode::ArmorDetectorNode() {
        detector_ = initDetector();
    }

    std::unique_ptr<Detector> ArmorDetectorNode::initDetector() {
        std::string model_path_armor = "../model/mobilenetv3_last_int_all_new/last.xml";
        float score_threshold = 0.7;
        float nms_threshold = 0.3;
        int detector_color = BLUE;

        auto detector = std::make_unique<Detector>(model_path_armor, score_threshold, nms_threshold, detector_color);
        std::string model_path_number = "../model/number_classifier.onnx";
        auto label_path = "../model/label.txt";
        double threshold = 0.7;
        std::vector<std::string> ignore_classes = {"negative"};
        detector->classifier =
                std::make_unique<NumberClassifier>(model_path_number, label_path, threshold, ignore_classes);

        return detector;
    }
} // namespace rm_auto_aim


int main() {
    rm_auto_aim::ArmorDetectorNode node;
    cv::VideoCapture capture("../test/test.mp4");

    if (!capture.isOpened()) {
        std::cout << "can not open video " << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        capture >> frame;
        if (frame.empty()) {
            std::cout << "read video end" << std::endl;
            break;
        }

        auto armors = node.detector_->detect(frame);

        if (!armors.empty()) {
            auto all_num_img = node.detector_->getAllNumbersImage();
        }

        // node.detector_->drawResults(frame);

        int k = cv::waitKey(10);
        if (k == 27) {
            std::cout << "退出" << std::endl;
            break;
        }
        cv::imshow("result", frame);
    }

    return 0;
}
