#include "errors.hpp"
#include "native.hpp"

#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

PUBLISH(yolo26, preprocess, std::vector<float>, std::span<const uint8_t> image) {
    constexpr int SIZE = 640;

    if (image.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        Mln::raise(Yolo26::Errors::Preprocess::Image::Too::Large, image.size(),
                   std::numeric_limits<int>::max());
    }

    cv::Mat encoded(1, static_cast<int>(image.size()), CV_8UC1,
                    const_cast<uint8_t *>(image.data()));
    cv::Mat bgr = cv::imdecode(encoded, cv::IMREAD_COLOR);
    if (bgr.empty())
        Mln::raise(Yolo26::Errors::Preprocess::Image::Decode::Failed);

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(SIZE, SIZE), 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    cv::Mat normalized;
    rgb.convertTo(normalized, CV_32FC3, 1.0 / 255.0);

    std::vector<float> out(3 * SIZE * SIZE);
    const std::size_t plane_size = static_cast<std::size_t>(SIZE) * SIZE;
    std::vector<cv::Mat> planes;
    planes.reserve(3);
    for (int channel = 0; channel < 3; ++channel)
        planes.emplace_back(SIZE, SIZE, CV_32FC1, out.data() + channel * plane_size);
    cv::split(normalized, planes);

    return out;
}
