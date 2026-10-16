#pragma once
#include <cstdint>
#include <mln/sdk/module.hpp>
#include <mln/sdk/tensor.hpp>

namespace Yolo26::Nn {
    class C3k2 : public Mln::Sdk::Module {
        public:
            C3k2(int64_t c1, int64_t c2, int64_t n = 1, bool c3k = false, float e = 0.5f,
                 bool add = true);
            Mln::Sdk::Tensor forward(const Mln::Sdk::Tensor &x);

        private:
            int64_t hidden_ch_;
            int64_t n_;
            int64_t c2_;
            bool c3k_;
            bool add_;
    };
}
