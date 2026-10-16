#pragma once
#include <cstdint>
#include <mln/sdk/module.hpp>
#include <mln/sdk/tensor.hpp>

namespace Yolo26::Nn {
    class Bottleneck : public Mln::Sdk::Module {
        public:
            Bottleneck(int64_t c1, int64_t c2, bool add = true, float e = 0.5f);
            Mln::Sdk::Tensor forward(const Mln::Sdk::Tensor &x);

        private:
            bool add_;
    };
}
