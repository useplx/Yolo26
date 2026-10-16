#include "nn/c3k2psa.hpp"
#include "nn/bottleneck.hpp"
#include "nn/conv.hpp"
#include "nn/psablock.hpp"
#include <cstdint>
#include <mln/sdk/functional.hpp>
#include <mln/sdk/modules.hpp>
#include <mln/sdk/operations.hpp>
#include <vector>

using namespace Mln::Sdk;

namespace Yolo26::Nn {
    C3k2PSA::C3k2PSA(int64_t c1, int64_t c2, float e) {
        name = "c3k2psa";
        hidden_ch_ = static_cast<int64_t>(c2 * e);

        modules.insert<Nn::Conv>("cv1", c1, 2 * hidden_ch_, 1, 1, 0);
        modules.insert<Nn::Conv>("cv2", 3 * hidden_ch_, c2, 1, 1, 0);
        modules.insert<Nn::Bottleneck>("m.0.0", hidden_ch_, hidden_ch_, true, 0.5f);
        modules.insert<Nn::PSABlock>("m.0.1", hidden_ch_);
    }

    Tensor C3k2PSA::forward(const Tensor &x) {
        Tensor mid = modules.at<Nn::Conv>("cv1")->forward(x);

        int64_t N = mid.shape[0], C = mid.shape[1], H = mid.shape[2], W = mid.shape[3];
        int64_t Ch = hidden_ch_;
        std::vector<int64_t> ones4 = {1, 1, 1, 1};

        Tensor chunk0 = operations.insert<Functional::Slice>(
            mid, std::vector<int64_t>{0, 0, 0, 0}, std::vector<int64_t>{N, Ch, H, W}, ones4);
        Tensor chunk1 = operations.insert<Functional::Slice>(
            mid, std::vector<int64_t>{0, Ch, 0, 0}, std::vector<int64_t>{N, C, H, W}, ones4);

        Tensor b = modules.at<Nn::Bottleneck>("m.0.0")->forward(chunk1);
        b = modules.at<Nn::PSABlock>("m.0.1")->forward(b);

        Tensor cat = operations.insert<Functional::Concatenate>(chunk0, chunk1, int64_t{1});
        cat = operations.insert<Functional::Concatenate>(cat, b, int64_t{1});

        return modules.at<Nn::Conv>("cv2")->forward(cat);
    }
}
