#pragma once
#include <cstdint>
#include <mln/sdk/module.hpp>
#include <mln/sdk/tensor.hpp>

namespace Yolo26::Nn {
    class C3k : public Mln::Sdk::Module {
        public:
            C3k(int64_t c1, int64_t c2, int64_t n = 2, bool shortcut = true, float e = 0.5f);
            Mln::Sdk::Tensor forward(const Mln::Sdk::Tensor &x);

        private:
            int64_t n_;
    };
}
