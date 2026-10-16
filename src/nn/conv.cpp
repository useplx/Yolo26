#include "nn/conv.hpp"
#include <mln/sdk/modules.hpp>

using namespace Mln::Sdk;

namespace Yolo26::Nn {
    Conv::Conv(int64_t in_ch, int64_t out_ch, int64_t k, int64_t s, int64_t p, int64_t g, bool act)
        : act_(act) {
        name = "conv";
        modules.insert<Mln::Sdk::Nn::Conv2D>("conv", in_ch, out_ch, k, s, p, 1, g);
        if (act_)
            modules.insert<Mln::Sdk::Nn::SiLU>("act");
    }

    Tensor Conv::forward(const Tensor &x) {
        auto y = modules.at<Mln::Sdk::Nn::Conv2D>("conv")->forward(x);
        if (act_)
            return modules.at<Mln::Sdk::Nn::SiLU>("act")->forward(y);
        return y;
    }
}
