#pragma once
#include <cstdint>
#include <mln/sdk/module.hpp>
#include <mln/sdk/tensor.hpp>

namespace Yolo26::Nn {

    class PSABlock : public Mln::Sdk::Module {
        public:
            explicit PSABlock(int64_t c, bool shortcut = true);
            Mln::Sdk::Tensor forward(const Mln::Sdk::Tensor &x);

        private:
            bool add_;
    };
}
