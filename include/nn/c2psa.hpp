#pragma once
#include <cstdint>
#include <mln/sdk/module.hpp>
#include <mln/sdk/tensor.hpp>

namespace Yolo26::Nn {
    class C2PSA : public Mln::Sdk::Module {
        public:
            explicit C2PSA(int64_t c, int64_t n = 1, float e = 0.5f);
            Mln::Sdk::Tensor forward(const Mln::Sdk::Tensor &x);

        private:
            int64_t c_, n_;
    };
}
