#pragma once
#include <cstdint>
#include <mln/sdk/module.hpp>
#include <mln/sdk/tensor.hpp>

namespace Yolo26::Nn {
    class C3k2PSA : public Mln::Sdk::Module {
        public:
            C3k2PSA(int64_t c1, int64_t c2, float e = 0.5f);
            Mln::Sdk::Tensor forward(const Mln::Sdk::Tensor &x);

        private:
            int64_t hidden_ch_;
    };
}
