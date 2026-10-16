#include "orchestration.hpp"
#include "native.hpp"
#include "result.hpp"

#include <algorithm>
#include <array>
#include <cmath>

using namespace Mln::Sdk;

namespace Yolo26 {
    namespace {
        constexpr int ANCHORS = 8400;
        constexpr int CLASSES = 80;
        constexpr float CONFIDENCE = 0.25f;
        constexpr int STRIDES[3] = {8, 16, 32};
        constexpr int IMG_SIZE = 640;

        constexpr std::array<const char *, CLASSES> COCO_LABELS = {
            "person",        "bicycle",      "car",
            "motorcycle",    "airplane",     "bus",
            "train",         "truck",        "boat",
            "traffic light", "fire hydrant", "stop sign",
            "parking meter", "bench",        "bird",
            "cat",           "dog",          "horse",
            "sheep",         "cow",          "elephant",
            "bear",          "zebra",        "giraffe",
            "backpack",      "umbrella",     "handbag",
            "tie",           "suitcase",     "frisbee",
            "skis",          "snowboard",    "sports ball",
            "kite",          "baseball bat", "baseball glove",
            "skateboard",    "surfboard",    "tennis racket",
            "bottle",        "wine glass",   "cup",
            "fork",          "knife",        "spoon",
            "bowl",          "banana",       "apple",
            "sandwich",      "orange",       "broccoli",
            "carrot",        "hot dog",      "pizza",
            "donut",         "cake",         "chair",
            "couch",         "potted plant", "bed",
            "dining table",  "toilet",       "tv",
            "laptop",        "mouse",        "remote",
            "keyboard",      "cell phone",   "microwave",
            "oven",          "toaster",      "sink",
            "refrigerator",  "book",         "clock",
            "vase",          "scissors",     "teddy bear",
            "hair drier",    "toothbrush"};

        struct Anchor {
                float x, y, stride;
        };

        Anchor anchor_at(int index) {
            int idx = index;
            for (int stride : STRIDES) {
                int wh = IMG_SIZE / stride;
                int count = wh * wh;
                if (idx < count) {
                    int row = idx / wh;
                    int col = idx % wh;
                    return {col + 0.5f, row + 0.5f, static_cast<float>(stride)};
                }
                idx -= count;
            }
            return {0.0f, 0.0f, 0.0f};
        }

        float sigmoid(float x) {
            float clamped = std::clamp(x, -60.0f, 60.0f);
            return 1.0f / (1.0f + std::exp(-clamped));
        }
    }

    Preprocess::Preprocess() { name = "preprocess"; }

    IMPORT(yolo26, preprocess, Mln::Buffer(Mln::Buffer));

    Mln::Buffer Preprocess::forward(Mln::Runtime::Session &, Mln::Buffer input) {
        return preprocess(std::move(input));
    }

    Postprocess::Postprocess() { name = "postprocess"; }

    Mln::Buffer Postprocess::forward(Mln::Runtime::Session &, Mln::Buffer raw) {
        std::vector<float> data(raw.size() / sizeof(float));
        raw.read(std::as_writable_bytes(std::span(data)));

        Detections detections;
        for (int i = 0; i < ANCHORS; i++) {
            float l = data[0 * ANCHORS + i];
            float t = data[1 * ANCHORS + i];
            float r = data[2 * ANCHORS + i];
            float b = data[3 * ANCHORS + i];

            int best_cls = 0;
            float best_score = -1.0f;
            for (int c = 0; c < CLASSES; c++) {
                float score = sigmoid(data[(4 + c) * ANCHORS + i]);
                if (score > best_score) {
                    best_score = score;
                    best_cls = c;
                }
            }
            if (best_score <= CONFIDENCE)
                continue;

            Anchor a = anchor_at(i);
            detections.push_back({COCO_LABELS[best_cls], best_score, (a.x - l) * a.stride,
                                  (a.y - t) * a.stride, (a.x + r) * a.stride,
                                  (a.y + b) * a.stride});
        }

        std::sort(detections.begin(), detections.end(),
                  [](const Detection &left, const Detection &right) {
                      return left.confidence > right.confidence;
                  });

        Mln::Sdk::Native::Return<void>::Writer writer;
        writer.write(detections);
        Mln::Buffer output(writer.size());
        output.write(writer.bytes());
        return output;
    }

    Orchestration::Orchestration() {
        name = "yolo26";

        modules.insert<::Yolo26::Yolo26>("model");

        orchestrations.insert<Preprocess>("preprocess");
        orchestrations.insert<Postprocess>("postprocess");
    }

    Mln::Buffer Orchestration::forward(Mln::Runtime::Session &session, Mln::Buffer input) {
        Mln::Buffer tensor =
            orchestrations.at<Preprocess>("preprocess")->forward(session, std::move(input));
        Mln::Buffer raw = session.invoke("yolo26.model", std::move(tensor));
        return orchestrations.at<Postprocess>("postprocess")->forward(session, std::move(raw));
    }
}
