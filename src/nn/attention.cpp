#include "nn/attention.hpp"
#include "nn/conv.hpp"
#include <cmath>
#include <cstdint>
#include <mln/sdk/functional.hpp>
#include <mln/sdk/modules.hpp>
#include <mln/sdk/operations.hpp>
#include <vector>

using namespace Mln::Sdk;

namespace Yolo26::Nn {
    Attention::Attention(int64_t dim) : dim_(dim) {
        name = "attn";
        num_heads_ = dim / 64;
        head_dim_ = dim / num_heads_;
        key_dim_ = static_cast<int64_t>(head_dim_ * 0.5);
        int64_t key_channels = key_dim_ * num_heads_;
        int64_t projection_channels = dim + key_channels * 2;
        scale_ = std::pow(static_cast<double>(key_dim_), -0.5);

        modules.insert<Nn::Conv>("qkv", dim, projection_channels, 1, 1, 0, 1, false);
        modules.insert<Nn::Conv>("proj", dim, dim, 1, 1, 0, 1, false);
        modules.insert<Nn::Conv>("pe", dim, dim, 3, 1, -1, dim, false);
    }

    Tensor Attention::forward(const Tensor &x) {
        int64_t batch = x.shape[0];
        int64_t channels = x.shape[1];
        int64_t height = x.shape[2];
        int64_t width = x.shape[3];
        int64_t tokens = height * width;
        int64_t c_per_head = key_dim_ * 2 + head_dim_;
        std::vector<int64_t> ones4 = {1, 1, 1, 1};

        Tensor qkv = modules.at<Nn::Conv>("qkv")->forward(x);
        qkv = operations.insert<Functional::Reshape>(
            qkv, std::vector<int64_t>{batch, num_heads_ * c_per_head, tokens});
        qkv = operations.insert<Functional::Reshape>(
            qkv, std::vector<int64_t>{batch, num_heads_, c_per_head, tokens});

        Tensor q = operations.insert<Functional::Slice>(
            qkv, std::vector<int64_t>{0, 0, 0, 0},
            std::vector<int64_t>{batch, num_heads_, key_dim_, tokens}, ones4);
        Tensor k = operations.insert<Functional::Slice>(
            qkv, std::vector<int64_t>{0, 0, key_dim_, 0},
            std::vector<int64_t>{batch, num_heads_, 2 * key_dim_, tokens}, ones4);
        Tensor v = operations.insert<Functional::Slice>(
            qkv, std::vector<int64_t>{0, 0, 2 * key_dim_, 0},
            std::vector<int64_t>{batch, num_heads_, c_per_head, tokens}, ones4);

        Tensor transposed_q =
            operations.insert<Functional::Transpose>(q, std::vector<int64_t>{0, 1, 3, 2});
        Tensor scores = operations.insert<Functional::MatMul>(transposed_q, k);

        Weight scale_weight{"", {}, scale_};
        Tensor scale_tensor = operations.insert<Functional::Const>(scale_weight, true);
        Tensor broadcast_scale = operations.insert<Functional::Broadcast>(
            scale_tensor, scores.shape, std::vector<int64_t>{});
        Tensor scaled = operations.insert<Functional::Multiply>(scores, broadcast_scale);

        Tensor exponentials = operations.insert<Functional::Exp>(scaled);
        Tensor denominator =
            operations.insert<Functional::ReduceSum>(exponentials, std::vector<int64_t>{3});
        Tensor reshaped_denominator = operations.insert<Functional::Reshape>(
            denominator, std::vector<int64_t>{batch, num_heads_, tokens, 1});
        Tensor broadcast_denominator = operations.insert<Functional::Broadcast>(
            reshaped_denominator, exponentials.shape, std::vector<int64_t>{0, 1, 2, 3});
        Tensor attention =
            operations.insert<Functional::Divide>(exponentials, broadcast_denominator);

        Tensor transposed_attention =
            operations.insert<Functional::Transpose>(attention, std::vector<int64_t>{0, 1, 3, 2});
        Tensor out = operations.insert<Functional::MatMul>(v, transposed_attention);

        Tensor spatial_output = operations.insert<Functional::Reshape>(
            out, std::vector<int64_t>{batch, channels, height, width});
        Tensor spatial_v = operations.insert<Functional::Reshape>(
            v, std::vector<int64_t>{batch, channels, height, width});
        Tensor positional_encoding = modules.at<Nn::Conv>("pe")->forward(spatial_v);

        Tensor combined = operations.insert<Functional::Add>(spatial_output, positional_encoding);
        return modules.at<Nn::Conv>("proj")->forward(combined);
    }
}
