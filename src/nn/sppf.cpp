#include "nn/sppf.hpp"
#include "nn/conv.hpp"
#include <cstdint>
#include <mln/sdk/functional.hpp>
#include <mln/sdk/modules.hpp>
#include <mln/sdk/operations.hpp>

using namespace Mln::Sdk;

namespace Yolo26::Nn {
    SPPF::SPPF(int64_t c1, int64_t c2, int64_t k, bool shortcut)
        : k_(k), half_ch_(c1 / 2), add_(shortcut && c1 == c2) {
        name = "sppf";
        modules.insert<Nn::Conv>("cv1", c1, half_ch_, 1, 1, 0, 1, false);
        modules.insert<Nn::Conv>("cv2", half_ch_ * 4, c2, 1, 1, 0);
    }

    Tensor SPPF::forward(const Tensor &x) {
        Tensor y = modules.at<Nn::Conv>("cv1")->forward(x);

        int64_t p = k_ / 2;
        std::vector<int64_t> kv = {k_, k_};
        std::vector<int64_t> sv = {1, 1};
        std::vector<int64_t> pv = {p, p, p, p};

        Tensor p1 = operations.insert<Functional::MaxPool2D>(y, kv, sv, pv);
        Tensor p2 = operations.insert<Functional::MaxPool2D>(p1, kv, sv, pv);
        Tensor p3 = operations.insert<Functional::MaxPool2D>(p2, kv, sv, pv);

        Tensor cat = operations.insert<Functional::Concatenate>(y, p1, int64_t{1});
        cat = operations.insert<Functional::Concatenate>(cat, p2, int64_t{1});
        cat = operations.insert<Functional::Concatenate>(cat, p3, int64_t{1});

        Tensor out = modules.at<Nn::Conv>("cv2")->forward(cat);
        if (add_)
            return operations.insert<Functional::Add>(out, x);
        return out;
    }
}
