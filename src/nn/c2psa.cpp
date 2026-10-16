#include "nn/c2psa.hpp"
#include "nn/conv.hpp"
#include "nn/psablock.hpp"
#include <cstdint>
#include <mln/sdk/functional.hpp>
#include <mln/sdk/modules.hpp>
#include <mln/sdk/operations.hpp>
#include <string>
#include <vector>

using namespace Mln::Sdk;

namespace Yolo26::Nn {
    C2PSA::C2PSA(int64_t c, int64_t n, float e) : n_(n) {
        name = "c2psa";
        c_ = static_cast<int64_t>(c * e);

        modules.insert<Nn::Conv>("cv1", c, 2 * c_, 1, 1, 0);
        modules.insert<Nn::Conv>("cv2", 2 * c_, c, 1, 1, 0);
        for (int64_t i = 0; i < n; i++)
            modules.insert<Nn::PSABlock>("m." + std::to_string(i), c_);
    }

    Tensor C2PSA::forward(const Tensor &x) {
        Tensor mid = modules.at<Nn::Conv>("cv1")->forward(x);

        int64_t N = mid.shape[0], H = mid.shape[2], W = mid.shape[3];
        std::vector<int64_t> ones4 = {1, 1, 1, 1};

        Tensor a = operations.insert<Functional::Slice>(mid, std::vector<int64_t>{0, 0, 0, 0},
                                                        std::vector<int64_t>{N, c_, H, W}, ones4);
        Tensor b = operations.insert<Functional::Slice>(
            mid, std::vector<int64_t>{0, c_, 0, 0}, std::vector<int64_t>{N, 2 * c_, H, W}, ones4);

        for (int64_t i = 0; i < n_; i++)
            b = modules.at<Nn::PSABlock>("m." + std::to_string(i))->forward(b);

        Tensor cat = operations.insert<Functional::Concatenate>(a, b, int64_t{1});
        return modules.at<Nn::Conv>("cv2")->forward(cat);
    }
}
