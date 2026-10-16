#include "nn/c3k2.hpp"
#include "nn/bottleneck.hpp"
#include "nn/c3k.hpp"
#include "nn/conv.hpp"
#include <cstdint>
#include <mln/sdk/functional.hpp>
#include <mln/sdk/modules.hpp>
#include <mln/sdk/operations.hpp>
#include <string>
#include <vector>

using namespace Mln::Sdk;

namespace Yolo26::Nn {
    C3k2::C3k2(int64_t c1, int64_t c2, int64_t n, bool c3k, float e, bool add)
        : n_(n), c2_(c2), c3k_(c3k), add_(add) {
        name = "c3k2";
        hidden_ch_ = static_cast<int64_t>(c2 * e);

        modules.insert<Nn::Conv>("cv1", c1, 2 * hidden_ch_, 1, 1, 0);

        for (int64_t i = 0; i < n; i++) {
            std::string mi = "m." + std::to_string(i);
            if (c3k_)
                modules.insert<Nn::C3k>(mi, hidden_ch_, hidden_ch_, 2, add_);
            else
                modules.insert<Nn::Bottleneck>(mi, hidden_ch_, hidden_ch_, add_);
        }

        modules.insert<Nn::Conv>("cv2", (2 + n) * hidden_ch_, c2, 1, 1, 0);
    }

    Tensor C3k2::forward(const Tensor &x) {

        Tensor mid = modules.at<Nn::Conv>("cv1")->forward(x);

        int64_t N = mid.shape[0];
        int64_t C = mid.shape[1];
        int64_t H = mid.shape[2];
        int64_t W = mid.shape[3];
        int64_t Ch = hidden_ch_;

        std::vector<int64_t> all1 = {1, 1, 1, 1};
        Tensor chunk0 = operations.insert<Functional::Slice>(
            mid, std::vector<int64_t>{0, 0, 0, 0}, std::vector<int64_t>{N, Ch, H, W}, all1);
        Tensor chunk1 = operations.insert<Functional::Slice>(
            mid, std::vector<int64_t>{0, Ch, 0, 0}, std::vector<int64_t>{N, C, H, W}, all1);

        Tensor last = chunk1;
        std::vector<Tensor> to_cat = {chunk0, chunk1};
        for (int64_t i = 0; i < n_; i++) {
            std::string mi = "m." + std::to_string(i);
            last = c3k_ ? modules.at<Nn::C3k>(mi)->forward(last)
                        : modules.at<Nn::Bottleneck>(mi)->forward(last);
            to_cat.push_back(last);
        }

        Tensor cat = to_cat[0];
        for (std::size_t i = 1; i < to_cat.size(); i++)
            cat = operations.insert<Functional::Concatenate>(cat, to_cat[i], int64_t{1});

        return modules.at<Nn::Conv>("cv2")->forward(cat);
    }
}
