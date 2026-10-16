#pragma once
#include <cstdint>
#include <mln/sdk/module.hpp>
#include <mln/sdk/tensor.hpp>

namespace Yolo26::Nn {
    class Attention : public Mln::Sdk::Module {
        public:
            explicit Attention(int64_t dim);
            Mln::Sdk::Tensor forward(const Mln::Sdk::Tensor &x);

        private:
            int64_t dim_, num_heads_, head_dim_, key_dim_;
            double scale_;
    };
}
