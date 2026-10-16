#include "nn/psablock.hpp"
#include "nn/attention.hpp"
#include "nn/conv.hpp"
#include <cstdint>
#include <mln/sdk/functional.hpp>
#include <mln/sdk/modules.hpp>
#include <mln/sdk/operations.hpp>

using namespace Mln::Sdk;

namespace Yolo26::Nn {
    PSABlock::PSABlock(int64_t c, bool shortcut) : add_(shortcut) {
        name = "psablock";
        modules.insert<Nn::Attention>("attn", c);
        modules.insert<Nn::Conv>("ffn.0", c, 2 * c, 1, 1, 0, 1, true);
        modules.insert<Nn::Conv>("ffn.1", 2 * c, c, 1, 1, 0, 1, false);
    }

    Tensor PSABlock::forward(const Tensor &x) {
        Tensor a = modules.at<Nn::Attention>("attn")->forward(x);
        Tensor x1 = add_ ? operations.insert<Functional::Add>(x, a) : a;

        Tensor f = modules.at<Nn::Conv>("ffn.0")->forward(x1);
        f = modules.at<Nn::Conv>("ffn.1")->forward(f);
        return add_ ? operations.insert<Functional::Add>(x1, f) : f;
    }
}
