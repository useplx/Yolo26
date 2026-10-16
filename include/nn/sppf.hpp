#pragma once
#include <cstdint>
#include <mln/sdk/module.hpp>
#include <mln/sdk/tensor.hpp>

namespace Yolo26::Nn {
    class SPPF : public Mln::Sdk::Module {
        public:
            SPPF(int64_t c1, int64_t c2, int64_t k = 5, bool shortcut = false);
            Mln::Sdk::Tensor forward(const Mln::Sdk::Tensor &x);

        private:
            int64_t k_;
            int64_t half_ch_;
            bool add_;
    };
}
