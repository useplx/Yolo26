#include <cstdint>

#include "nn/bottleneck.hpp"
#include "nn/conv.hpp"

#include <mln/sdk/functional.hpp>
#include <mln/sdk/modules.hpp>
#include <mln/sdk/operations.hpp>

using namespace Mln::Sdk;

namespace Yolo26::Nn {
    Bottleneck::Bottleneck(int64_t c1, int64_t c2, bool add, float e) : add_(add) {
        name = "bottleneck";
        int64_t hidden = static_cast<int64_t>(c2 * e);
        modules.insert<Nn::Conv>("cv1", c1, hidden, 3, 1);
        modules.insert<Nn::Conv>("cv2", hidden, c2, 3, 1);
    }

    Tensor Bottleneck::forward(const Tensor &x) {
        Tensor y = modules.at<Nn::Conv>("cv2")->forward(modules.at<Nn::Conv>("cv1")->forward(x));
        if (add_)
            return operations.insert<Functional::Add>(x, y);
        return y;
    }
}
