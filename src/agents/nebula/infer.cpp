// run the trained model on CPU as an independent agent, converting observations into features and returning actions.
#include "features.hpp"
#include "engine/protocol.hpp"
#ifdef _MSC_VER
// these diagnostics originate in LibTorch's schema and iterator templates.
#pragma warning(push)
#pragma warning(disable: 4267 4702)
#endif
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/csrc/jit/serialization/import.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "Usage: nebula MODEL.pt\n"; return 1; }
    try {
        std::ios::sync_with_stdio(false);
        std::cin.tie(nullptr);
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        auto model = torch::jit::load(argv[1], at::kCPU);
        model.eval();
        at::NoGradGuard inference;
        int side = static_cast<int>(model.attr("side").toInt());
        auto init = Protocol::readInit(std::cin);
        if (!init) throw std::invalid_argument("Invalid initialization");
        Learning::Features memory(side);
        while (std::cin.peek() != std::char_traits<char>::eof()) {
            auto view = Protocol::readObservation(std::cin, *init);
            if (!view) throw std::invalid_argument("Invalid observation");
            memory.update(*view);
            auto board = at::from_blob(memory.spatial.data(), {1, Learning::Channels, side, side}, at::kFloat);
            // quantization matches the training rollout before normalization inside the model.
            board = board.to(at::kBFloat16).to(at::kFloat);
            auto legal = at::from_blob(memory.legal.data(), {1, Learning::Actions, side, side}, at::kByte).to(at::kBool);
            auto history = at::from_blob(memory.temporal.data(), {1, 2, Learning::Window}, at::kFloat);
            auto result = model.forward({board, legal, history}).toTuple();
            int index = result->elements()[0].toTensor().argmax(1).item<int>();
            auto action = Learning::decode(index, side);
            if (!Protocol::writeAction(std::cout, action) || !(std::cout << std::flush)) return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "nebula: " << error.what() << '\n';
        return 1;
    }
}
