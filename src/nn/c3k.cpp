#include "nn/c3k.hpp"
#include "nn/bottleneck.hpp"
#include "nn/conv.hpp"
#include <cstdint>
#include <mln/sdk/functional.hpp>
#include <mln/sdk/modules.hpp>
#include <mln/sdk/operations.hpp>
#include <string>

using namespace Mln::Sdk;

namespace Yolo26::Nn {
    C3k::C3k(int64_t c1, int64_t c2, int64_t n, bool shortcut, float e) : n_(n) {
        name = "c3k";
        int64_t c_ = static_cast<int64_t>(c2 * e);

        modules.insert<Nn::Conv>("cv1", c1, c_, 1, 1, 0);
        modules.insert<Nn::Conv>("cv2", c1, c_, 1, 1, 0);
        modules.insert<Nn::Conv>("cv3", 2 * c_, c2, 1, 1, 0);

        for (int64_t i = 0; i < n; i++) {
            modules.insert<Nn::Bottleneck>("m." + std::to_string(i), c_, c_, shortcut, 1.0f);
        }
    }

    Tensor C3k::forward(const Tensor &x) {
        Tensor a = modules.at<Nn::Conv>("cv1")->forward(x);
        for (int64_t i = 0; i < n_; i++)
            a = modules.at<Nn::Bottleneck>("m." + std::to_string(i))->forward(a);

        Tensor b = modules.at<Nn::Conv>("cv2")->forward(x);

        Tensor cat = operations.insert<Functional::Concatenate>(a, b, int64_t{1});
        return modules.at<Nn::Conv>("cv3")->forward(cat);
    }
}
