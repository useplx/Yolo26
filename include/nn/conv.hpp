#pragma once
#include <cstdint>
#include <mln/sdk/module.hpp>
#include <mln/sdk/nn/conv2d.hpp>
#include <mln/sdk/nn/silu.hpp>
#include <mln/sdk/tensor.hpp>

namespace Yolo26::Nn {
    class Conv : public Mln::Sdk::Module {
        public:
            Conv(int64_t in_ch, int64_t out_ch, int64_t k = 1, int64_t s = 1, int64_t p = -1,
                 int64_t g = 1, bool act = true);

            Mln::Sdk::Tensor forward(const Mln::Sdk::Tensor &x);

        private:
            bool act_;
    };
}
