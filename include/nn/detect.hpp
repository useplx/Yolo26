#pragma once
#include <cstdint>
#include <mln/sdk/module.hpp>
#include <mln/sdk/tensor.hpp>
#include <vector>

namespace Yolo26::Nn {
    class Detect : public Mln::Sdk::Module {
        public:
            Detect(int64_t nc, std::vector<int64_t> ch);

            Mln::Sdk::Tensor forward(const std::vector<Mln::Sdk::Tensor> &x);

        private:
            int64_t nc_;
            int64_t c2_, c3_;
            std::vector<int64_t> ch_;
    };
}
