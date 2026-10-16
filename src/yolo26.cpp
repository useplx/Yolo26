#include "yolo26.hpp"
#include "errors.hpp"
#include "nn/c2psa.hpp"
#include "nn/c3k2.hpp"
#include "nn/c3k2psa.hpp"
#include "nn/conv.hpp"
#include "nn/detect.hpp"
#include "nn/sppf.hpp"
#include <mln/sdk/functional.hpp>
#include <mln/sdk/modules.hpp>
#include <mln/sdk/operations.hpp>

using namespace Mln::Sdk;

namespace Yolo26 {
    Yolo26::Yolo26(int64_t nc) : nc_(nc) {
        name = "model";

        modules.insert<Nn::Conv>("0", 3, 16, 3, 2);
        modules.insert<Nn::Conv>("1", 16, 32, 3, 2);
        modules.insert<Nn::C3k2>("2", 32, 64, 1, false, 0.25f);
        modules.insert<Nn::Conv>("3", 64, 64, 3, 2);
        modules.insert<Nn::C3k2>("4", 64, 128, 1, false, 0.25f);
        modules.insert<Nn::Conv>("5", 128, 128, 3, 2);
        modules.insert<Nn::C3k2>("6", 128, 128, 1, true, 0.5f);
        modules.insert<Nn::Conv>("7", 128, 256, 3, 2);
        modules.insert<Nn::C3k2>("8", 256, 256, 1, true, 0.5f);
        modules.insert<Nn::SPPF>("9", 256, 256, 5, true);
        modules.insert<Nn::C2PSA>("10", 256, 1, 0.5f);

        modules.insert<Nn::C3k2>("13", 384, 128, 1, true, 0.5f);

        modules.insert<Nn::C3k2>("16", 256, 64, 1, true, 0.5f);

        modules.insert<Nn::Conv>("17", 64, 64, 3, 2);

        modules.insert<Nn::C3k2>("19", 192, 128, 1, true, 0.5f);
        modules.insert<Nn::Conv>("20", 128, 128, 3, 2);

        modules.insert<Nn::C3k2PSA>("22", 384, 256, 0.5f);

        modules.insert<Nn::Detect>("23", nc, std::vector<int64_t>{64, 128, 256});
    }

    std::vector<Tensor> Yolo26::graph(std::span<const Tensor> inputs) {
        if (inputs.size() != 1)
            Mln::raise(Errors::Graph::Input::Count::Invalid, inputs.size());
        return {forward(inputs.front())};
    }

    std::vector<std::vector<Tensor>> Yolo26::specializations() const {
        return {{Tensor({1, 3, 640, 640}, Mln::DType::Float32)}};
    }

    Tensor Yolo26::forward(Tensor x) {

        x = modules.at<Nn::Conv>("0")->forward(x);
        x = modules.at<Nn::Conv>("1")->forward(x);
        x = modules.at<Nn::C3k2>("2")->forward(x);
        x = modules.at<Nn::Conv>("3")->forward(x);
        Tensor p3 = modules.at<Nn::C3k2>("4")->forward(x);
        x = modules.at<Nn::Conv>("5")->forward(p3);
        Tensor p4 = modules.at<Nn::C3k2>("6")->forward(x);
        x = modules.at<Nn::Conv>("7")->forward(p4);
        x = modules.at<Nn::C3k2>("8")->forward(x);
        x = modules.at<Nn::SPPF>("9")->forward(x);
        Tensor p5 = modules.at<Nn::C2PSA>("10")->forward(x);

        Tensor up5 = operations.insert<Functional::Upsample>(p5, std::vector<int64_t>{2, 2});
        Tensor cat54 = operations.insert<Functional::Concatenate>(up5, p4, int64_t{1});
        Tensor n4 = modules.at<Nn::C3k2>("13")->forward(cat54);

        Tensor up4 = operations.insert<Functional::Upsample>(n4, std::vector<int64_t>{2, 2});
        Tensor cat43 = operations.insert<Functional::Concatenate>(up4, p3, int64_t{1});
        Tensor n3 = modules.at<Nn::C3k2>("16")->forward(cat43);

        Tensor dn3 = modules.at<Nn::Conv>("17")->forward(n3);
        Tensor cat3 = operations.insert<Functional::Concatenate>(dn3, n4, int64_t{1});
        Tensor n4h = modules.at<Nn::C3k2>("19")->forward(cat3);

        Tensor dn4 = modules.at<Nn::Conv>("20")->forward(n4h);
        Tensor cat4 = operations.insert<Functional::Concatenate>(dn4, p5, int64_t{1});
        Tensor n5h = modules.at<Nn::C3k2PSA>("22")->forward(cat4);

        return modules.at<Nn::Detect>("23")->forward({n3, n4h, n5h});
    }
}
