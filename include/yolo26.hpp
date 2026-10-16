#pragma once
#include "nn/detect.hpp"
#include <mln/sdk/module.hpp>
#include <mln/sdk/tensor.hpp>

namespace Yolo26 {
    class Yolo26 : public Mln::Sdk::Module {
        public:
            explicit Yolo26(int64_t nc = 80);
            std::vector<Mln::Sdk::Tensor> graph(std::span<const Mln::Sdk::Tensor> inputs) override;
            std::vector<std::vector<Mln::Sdk::Tensor>> specializations() const override;
            Mln::Sdk::Tensor forward(Mln::Sdk::Tensor x);

        private:
            int64_t nc_;
    };
}
