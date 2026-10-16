#pragma once

#include <mln/sdk/orchestration.hpp>

#include "yolo26.hpp"

namespace Yolo26 {

    class Preprocess : public Mln::Sdk::Orchestration {
        public:
            Preprocess();
            Mln::Buffer forward(Mln::Runtime::Session &session, Mln::Buffer input) override;
    };

    class Postprocess : public Mln::Sdk::Orchestration {
        public:
            Postprocess();
            Mln::Buffer forward(Mln::Runtime::Session &session, Mln::Buffer raw) override;
    };

    class Orchestration : public Mln::Sdk::Orchestration {
        public:
            Orchestration();
            Mln::Buffer forward(Mln::Runtime::Session &session, Mln::Buffer input) override;
    };
}
