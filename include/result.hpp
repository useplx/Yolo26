#pragma once

#include <mln/sdk/native/serialization.hpp>

#include <string>
#include <vector>

namespace Yolo26 {
    struct Detection {
            std::string label;
            float confidence = 0.0f;
            float x1 = 0.0f;
            float y1 = 0.0f;
            float x2 = 0.0f;
            float y2 = 0.0f;
    };

    using Detections = std::vector<Detection>;
}

namespace Mln::Sdk::Native {
    template <> struct Return<Yolo26::Detection> {
            static void write(Return<void>::Writer &writer, const Yolo26::Detection &detection) {
                writer.write_field(detection.label);
                writer.write(detection.confidence);
                writer.write(detection.x1);
                writer.write(detection.y1);
                writer.write(detection.x2);
                writer.write(detection.y2);
            }
    };

    template <> struct Argument<Yolo26::Detection> {
            static Yolo26::Detection read(Argument<void>::Reader &reader) {
                Yolo26::Detection detection;
                detection.label = reader.read_field<std::string>();
                detection.confidence = reader.read<float>();
                detection.x1 = reader.read<float>();
                detection.y1 = reader.read<float>();
                detection.x2 = reader.read<float>();
                detection.y2 = reader.read<float>();
                return detection;
            }
    };
}
