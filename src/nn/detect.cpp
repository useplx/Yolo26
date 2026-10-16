#include "nn/detect.hpp"
#include "nn/conv.hpp"
#include <algorithm>
#include <cstdint>
#include <mln/sdk/functional.hpp>
#include <mln/sdk/modules.hpp>
#include <mln/sdk/nn/conv2d.hpp>
#include <mln/sdk/operations.hpp>
#include <string>

using namespace Mln::Sdk;

namespace Yolo26::Nn {

    Detect::Detect(int64_t nc, std::vector<int64_t> ch) : nc_(nc), ch_(ch) {
        name = "detect";
        c2_ = std::max<int64_t>({16, ch[0] / 4, 4});
        c3_ = std::max<int64_t>({ch[0], std::min<int64_t>(nc, 100)});

        for (std::size_t i = 0; i < ch.size(); i++) {
            int64_t c_in = ch[i];
            std::string si = std::to_string(i);

            modules.insert<Nn::Conv>("one2one_cv2." + si + ".0", c_in, c2_, 3, 1);
            modules.insert<Nn::Conv>("one2one_cv2." + si + ".1", c2_, c2_, 3, 1);
            modules.insert<Mln::Sdk::Nn::Conv2D>("one2one_cv2." + si + ".2", c2_, 4, 1, 1, 0, 1, 1,
                                                 true);

            modules.insert<Nn::Conv>("one2one_cv3." + si + ".0.0", c_in, c_in, 3, 1, -1, c_in);
            modules.insert<Nn::Conv>("one2one_cv3." + si + ".0.1", c_in, c3_, 1, 1, 0);
            modules.insert<Nn::Conv>("one2one_cv3." + si + ".1.0", c3_, c3_, 3, 1, -1, c3_);
            modules.insert<Nn::Conv>("one2one_cv3." + si + ".1.1", c3_, c3_, 1, 1, 0);
            modules.insert<Mln::Sdk::Nn::Conv2D>("one2one_cv3." + si + ".2", c3_, nc, 1, 1, 0, 1, 1,
                                                 true);
        }
    }

    Tensor Detect::forward(const std::vector<Tensor> &x) {
        Tensor result;
        bool first = true;

        for (std::size_t i = 0; i < ch_.size(); i++) {
            std::string si = std::to_string(i);
            const Tensor &xi = x[i];

            int64_t N = xi.shape[0], H = xi.shape[2], W = xi.shape[3];

            Tensor b = modules.at<Nn::Conv>("one2one_cv2." + si + ".0")->forward(xi);
            b = modules.at<Nn::Conv>("one2one_cv2." + si + ".1")->forward(b);
            b = modules.at<Mln::Sdk::Nn::Conv2D>("one2one_cv2." + si + ".2")->forward(b);

            b = operations.insert<Functional::Reshape>(b, std::vector<int64_t>{N, 4, H * W});

            Tensor c = modules.at<Nn::Conv>("one2one_cv3." + si + ".0.0")->forward(xi);
            c = modules.at<Nn::Conv>("one2one_cv3." + si + ".0.1")->forward(c);
            c = modules.at<Nn::Conv>("one2one_cv3." + si + ".1.0")->forward(c);
            c = modules.at<Nn::Conv>("one2one_cv3." + si + ".1.1")->forward(c);
            c = modules.at<Mln::Sdk::Nn::Conv2D>("one2one_cv3." + si + ".2")->forward(c);

            c = operations.insert<Functional::Reshape>(c, std::vector<int64_t>{N, nc_, H * W});

            Tensor scale_out = operations.insert<Functional::Concatenate>(b, c, int64_t{1});

            if (first) {
                result = scale_out;
                first = false;
            } else {
                result = operations.insert<Functional::Concatenate>(result, scale_out, int64_t{2});
            }
        }
        return result;
    }

}
